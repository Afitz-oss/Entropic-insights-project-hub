# api_request_processor.py
import aiohttp
import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field

# Define data classes for tracking request statuses and details
@dataclass
class StatusTracker:
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0

@dataclass
class APIRequest:
    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(self, request_url, request_header, retry_queue, save_filepath, status_tracker):
        logging.info(f"Starting request #{self.task_id}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url=request_url, headers=request_header, json=self.request_json) as response:
                    response_data = await response.json()
            if "error" in response_data:
                raise Exception(response_data["error"])
            self.result.append(response_data)
            append_to_jsonl([self.request_json, response_data, self.metadata], save_filepath)
            status_tracker.num_tasks_succeeded += 1
        except Exception as e:
            logging.warning(f"Request {self.task_id} failed with Exception {str(e)}")
            if self.attempts_left > 0:
                retry_queue.put_nowait(self)
            else:
                append_to_jsonl([self.request_json, {"error": str(e)}, self.metadata], save_filepath)
                status_tracker.num_tasks_failed += 1
        finally:
            status_tracker.num_tasks_in_progress -= 1

def append_to_jsonl(data, filename):
    with open(filename, "a") as f:
        f.write(json.dumps(data) + "\n")

async def process_api_requests_from_file(requests_filepath, save_filepath, request_url, api_key, max_requests_per_minute, max_tokens_per_minute, max_attempts, logging_level):
    logging.basicConfig(level=logging_level)
    request_header = {"Authorization": f"Bearer {api_key}"}
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = (i for i in range(1000000))  # Simplified task ID generator
    status_tracker = StatusTracker()

    with open(requests_filepath) as file:
        requests = (json.loads(line.strip()) for line in file)
        for request_json in requests:
            await asyncio.sleep(1 / max_requests_per_minute)  # Throttle request rate
            request = APIRequest(
                task_id=next(task_id_generator),
                request_json=request_json,
                token_consumption=len(request_json["input"]),  # Simplified token consumption estimation
                attempts_left=max_attempts,
                metadata=request_json.get("metadata", {})
            )
            status_tracker.num_tasks_started += 1
            status_tracker.num_tasks_in_progress += 1
            asyncio.create_task(request.call_api(request_url, request_header, queue_of_requests_to_retry, save_filepath, status_tracker))

        while status_tracker.num_tasks_in_progress > 0:
            await asyncio.sleep(0.1)  # Wait for all tasks to finish

        logging.info("All requests processed.")

def run_parallel_api_requests(filepath, results_path, api_key, request_url="https://api.openai.com/v1/embeddings"):
    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=filepath,
            save_filepath=results_path,
            request_url=request_url,
            api_key=api_key,
            max_requests_per_minute=1500,
            max_tokens_per_minute=6250000,
            max_attempts=5,
            logging_level=logging.INFO
        )
    )

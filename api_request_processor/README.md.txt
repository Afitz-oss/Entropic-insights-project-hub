# API Request Processor

`api_request_processor.py` is a Python script that efficiently processes multiple API requests in parallel using asynchronous programming. Originally designed to handle large-scale data with minimal manual intervention, this script reads requests from a file, sends them to an API endpoint, manages retries, and stores results in a structured output file. 

## Features

- **High Concurrency**: Processes multiple requests in parallel with asynchronous handling.
- **Throttling**: Limits requests per minute and tokens per minute to comply with API rate limits.
- **Error Handling**: Retries requests on failure and logs errors.
- **Data Persistence**: Saves results and errors in JSON Lines format for easy data management.

## Requirements

- Python 3.8+
- `aiohttp`, `asyncio`, `json`, `logging`, `pandas`, `openpyxl`

Install the required dependencies:

```bash
pip install -r requirements.txt

Usage
Setup API Key and Configuration: Copy config_example.json to config.json and update it with your API key and other parameters.

Prepare Input File: Create a .jsonl file with one request JSON object per line.

Run the Script:

from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, \
                        LLMPredictor, PromptHelper

from langchain.chat_models import ChatOpenAI 
import openai
import os

# Load API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

if not openai.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")


def index_documents(folder):
    max_input_size    = 4096
    num_outputs       = 512
    max_chunk_overlap = 100
    chunk_size_limit  = 600

    prompt_helper = PromptHelper(max_input_size, 
                                 num_outputs, 
                                 max_chunk_overlap = 0.5, 
                                 chunk_size_limit = chunk_size_limit)
    
    llm_predictor = LLMPredictor(
        llm = ChatOpenAI(temperature = 0.7, 
                         model_name = "gpt-3.5-turbo-0301", 
                         max_tokens = num_outputs)
        )

    documents = SimpleDirectoryReader(folder).load_data()

    index = GPTVectorStoreIndex.from_documents(
                documents, 
                llm_predictor = llm_predictor, 
                prompt_helper = prompt_helper)

    index.storage_context.persist(persist_dir=".") # save in current directory


index_documents(r"C:\Users\AkimFitzgerald\LLama index dir")

from llama_index import StorageContext, load_index_from_storage

def my_chatGPT_bot(input_text):
    # load the index from vector_store.json
    storage_context = StorageContext.from_defaults(persist_dir=".")
    index = load_index_from_storage(storage_context)

    # create a query engine to ask question
    query_engine = index.as_query_engine()
    response = query_engine.query(input_text)
    return response.response

my_chatGPT_bot('Tell me the summary of what happened in Borisov, Russia ')

print(my_chatGPT_bot('Tell me the summary of what happened in Borisov, Russia'))

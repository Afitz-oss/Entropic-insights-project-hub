import openai
import spacy
import logging
from sklearn.cluster import KMeans
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, \
                        LLMPredictor, PromptHelper, StorageContext
from langchain.chat_models import ChatOpenAI 
from collections import defaultdict
import pickle


# Setup logging
logging.basicConfig(filename='app.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get API key from environment variable
import os
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logging.error("API key not found in environment variables!")
    exit()

openai.api_key = openai_api_key


nlp = spacy.load("en_core_web_sm")

def index_documents(folder_path):
    """
    Indexes documents present in the provided folder using Llama Index and OpenAI's API.
    
    Parameters:
    - folder_path: Path to the directory containing documents to be indexed.

    Returns:
    - Saves the index to the current directory.
    """
    
    # Set parameters for document chunking
    max_input_size = 4096
    num_outputs = 512
    chunk_overlap_ratio = 100
    chunk_size_limit = 600
    
    # Create a helper for creating prompts
    prompt_helper = PromptHelper(
        max_input_size, 
        num_outputs, 
        chunk_overlap_ratio = 0.5, 
        chunk_size_limit = chunk_size_limit
    )
    
    # Set up a predictor using a language model
    llm_predictor = LLMPredictor(
        llm = ChatOpenAI(
            temperature = 0.7, 
            model_name = "gpt-3.5-turbo-0301", 
            max_tokens = num_outputs,
            openai_api_key = openai_api_key
        )
    )
    
    # Load documents from directory
    documents = SimpleDirectoryReader(folder_path).load_data()
    
    # Build an index from the loaded documents
    index = GPTVectorStoreIndex.from_documents(
        documents, 
        llm_predictor = llm_predictor, 
        prompt_helper = prompt_helper
    )

    # Save the index in the current directory
    index.storage_context.persist(persist_dir=".")
    print(f"Documents indexed and saved in the current directory!")

folder_path = r"C:\Users\AkimFitzgerald\Llama Dir"
index_documents(folder_path)


def get_embedding(text, model_name="text-embedding-ada-002"):
    """
    Fetches the embedding of the provided text using OpenAI's API.

    Parameters:
    - text (str): Text for which the embedding is required.
    - model_name (str, optional): The model to be used for extracting embeddings. 
                                  Defaults to "text-embedding-ada-002".

    Returns:
    - list: The embedding of the provided text.
    """
    
    # Ensure OpenAI API key is set
    if not openai.api_key:
        raise ValueError("OpenAI API key not set!")
    
    # Request the embedding
    response = openai.Embedding.create(
        model=model_name,
        input=[text]
    )
    
    # Extract the embedding from the response and return
    embedding = response['data'][0]['embedding']
    return embedding


def extract_entities(text):
    """
    Extracts named entities from the provided text using spaCy.

    Parameters:
    - text (str): Text from which entities are to be extracted.

    Returns:
    - list of tuples: Each tuple contains the entity and its type.
    """
    
    # Parse the text using spaCy's NLP model
    doc = nlp(text)
    
    # Extract entities and their corresponding types
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    return entities

def disambiguate_entity(entity, model_name="gpt-3.5-turbo"):
    """
    Uses OpenAI's API to provide clarity about an ambiguous entity.

    Parameters:
    - entity (str): The ambiguous entity that needs to be clarified.
    - model_name (str, optional): The model to be used for disambiguation. Defaults to "gpt-3.5-turbo".

    Returns:
    - str: A clearer or more detailed description of the entity.
    """
    
    # Ensure OpenAI API key is set
    if not openai.api_key:
        raise ValueError("OpenAI API key not set!")
    
    # Construct a prompt to ask the model for clarity
    prompt = f"Can you clarify who or what '{entity}' refers to?"
    
    # Obtain the model's response
    response = openai.Completion.create(
        model=model_name,
        prompt=prompt,
        max_tokens=2000
    )
    
    # Extract the model's answer from the response and return
    clarification = response.choices[0].text.strip()
    return clarification

def summarize_text(text, model_name="gpt-3.5-turbo", max_summary_tokens=2000):
    """
    Uses OpenAI's API to generate a summary of the provided text.

    Parameters:
    - text (str): Text that needs to be summarized.
    - model_name (str, optional): The model to be used for summarization. Defaults to "gpt-3.5-turbo".
    - max_summary_tokens (int, optional): The maximum number of tokens for the summary. Defaults to 200.

    Returns:
    - str: The summary of the provided text.
    """
    
    # Ensure OpenAI API key is set
    if not openai.api_key:
        raise ValueError("OpenAI API key not set!")
    
    # Construct a prompt to ask the model for a summary
    prompt = f"Provide a concise summary for the following text:\n{text}"
    
    # Obtain the model's response
    response = openai.Completion.create(
        model=model_name,
        prompt=prompt,
        max_tokens=max_summary_tokens
    )
    
    # Extract the model's summary from the response and return
    summary = response.choices[0].text.strip()
    return summary

def process_directory(directory_path):
    """
    Process all documents in the given directory:
    1. Fetches embeddings.
    2. Extracts entities.
    3. Disambiguates entities.
    4. Summarizes the text.
    
    Parameters:
    - directory_path (str): Path to the directory containing the documents.
    
    Returns:
    - dict: A dictionary where each key is a filename and the value is another dictionary containing the embedding, entities, disambiguated entities, and summary.
    """

    # Create a directory reader to fetch all document content
    dir_reader = SimpleDirectoryReader(directory_path)
    documents = dir_reader.load_data()

    results = {}
    all_entities = defaultdict(list)  # This will store all the found entities

    for doc_name, doc_content in documents.items():
        # 1. Get the embeddings for the document content
        embedding = get_embedding(doc_content)
        
        # 2. Extract entities
        entities = extract_entities(doc_content)
        
        # Add extracted entities to all_entities for later logging
        for entity, _ in entities:
            all_entities[entity].append(entity)
        
        # 3. Disambiguate entities
        disambiguated_entities = [disambiguate_entity(entity[0]) for entity in entities]
        
        # 4. Summarize the document
        summary = summarize_text(doc_content)

        results[doc_name] = {
            'embedding': embedding,
            'entities': entities,
            'disambiguated_entities': disambiguated_entities,
            'summary': summary
        }

    # Logging similar entities
    for entity, similar_entities in all_entities.items():
        if len(similar_entities) > 1:
            logging.info(f"Found similar names for '{entity}': {similar_entities}")

    return results

processed_data = process_directory(r"C:\Users\AkimFitzgerald\Llama Dir")
print(processed_data)



with open('processed_data.pkl', 'wb') as f:
    pickle.dump(processed_data, f)
    data = pickle.load(f)
print(data)
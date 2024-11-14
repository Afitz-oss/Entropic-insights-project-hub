# NLP Document Enrichment and Embedding

This project provides an NLP pipeline for enriching documents with named entity recognition, summarization, and embedding generation. The output includes structured information about each document, allowing for deeper insights and future analyses, such as clustering and similarity matching.

## Features
- **Named Entity Recognition (NER)**: Extracts and labels entities (like names and locations) using SpaCy.
- **Entity Disambiguation**: Clarifies ambiguous entities using OpenAI's language model.
- **Summarization**: Summarizes documents with OpenAI’s `gpt-3.5-turbo` model.
- **Embedding Generation**: Creates document embeddings for potential clustering and similarity analysis.

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/Document-Processing-and-Retrieval.git
   cd Document-Processing-and-Retrieval/NLP-Document-Enrichment

Install Requirements:
pip install -r requirements.txt

Configure API Key:

Copy config_example.json to config.json.
Add your OpenAI API key to the config.json file.

Download SpaCy Model:
python -m spacy download en_core_web_sm

Run the Enrichment Pipeline:
python document_enrichment.py

Output:

The processed data is saved as a .pkl file containing entities, disambiguated entities, summaries, and embeddings for each document.
Logs for entity similarities and processing details are saved in app.log.
Project Files
document_enrichment.py: Main script for document processing.
requirements.txt: Lists dependencies for the project.
config_example.json: Sample configuration for API key.
Future Enhancements
Add a clustering function for document similarity analysis.
Integrate additional NLP tasks, such as sentiment analysis.


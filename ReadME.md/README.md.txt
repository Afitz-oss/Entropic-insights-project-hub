Advanced Data Science, NLP, and Automation Projects
This repository showcases a diverse range of data science, NLP, and automation projects, from automating data entry tasks to building an AI-driven recipe suggestion app. Each project demonstrates unique techniques and approaches to tackle real-world challenges, including large-scale data processing, semantic search, and machine learning for personalized recommendations.

Table of Contents
Project Overview
Projects
Bulk Sponsor and Children Workbook Generator
NLP Document Enrichment
Semantic Search Chatbot
Name Matching Logistic Regression Model
Report Text Categorization
Recipe Generator and AI Meal Suggestion App
Setup
Future Enhancements
Contributing
License
Project Overview
This repository contains projects that demonstrate advanced data processing, NLP, and automation workflows, tackling tasks like document processing, semantic search, automated data categorization, and personalized meal recommendation. Each project is well-documented and organized for ease of use and future extensions.

Projects
Bulk Sponsor and Children Workbook Generator
Description: Automates the generation of structured Excel workbooks for sponsors and unreported children, reducing a six-month manual process to just 10 minutes. It processes over 50,000 files, extracting data from text files and PDFs, populating Excel templates, and handling data entry tasks that previously required 20 employees.
Key Features: Workbook creation, synthetic data generation, automated data entry, and validation.
Project Path: Bulk-Sponsor-Children-Workbook-Generator

NLP Document Enrichment
Description: Enriches documents with named entity recognition (NER), entity disambiguation, summarization, and embedding generation. This enables structured information extraction from text for downstream tasks, such as clustering and similarity matching.
Key Features: Named entity recognition, entity disambiguation, text summarization, and embedding generation.
Project Path: NLP-Document-Enrichment

Semantic Search Chatbot
Description: Implements a semantic search chatbot that allows users to query indexed documents using natural language. The chatbot uses embeddings for contextually accurate search responses, making it ideal for querying large document repositories.
Key Features: Document indexing, natural language querying, and interactive responses.
Project Path: Semantic-Search-Chatbot

Name Matching Logistic Regression Model
Description: Trains a logistic regression model to identify matching name pairs based on Jaro-Winkler and Levenshtein similarity scores, helping to identify and reconcile records with slight name variations.
Key Features: Similarity score calculation, binary classification, and performance evaluation using accuracy, precision, recall, and F1 score.
Project Path: Name-Matching-Logistic-Regression

Report Text Categorization
Description: Automates the categorization of report text data into three main topics: Politics and War Conflict, Human Rights and Social Issues, and Economic Impact and Infrastructure. It tokenizes and cleans the text, assigns categories based on keywords, and saves the categorized data as a new Excel file.
Key Features: Text processing, keyword-based categorization, and automated workflow.
Project Path: Report-Text-Categorization

Recipe Generator and AI Meal Suggestion App
Description: A web-based application built with Dash, which provides personalized meal suggestions using the OpenAI API and retrieves recipes from the Tasty API. Users can input dietary preferences, search for recipes, and receive AI-generated meal ideas tailored to their preferences.
Key Features: Personalized meal recommendations, recipe retrieval, and interactive user interface.
Project Path: Recipe-Generator-and-AI-Meal-Suggestion-App

Setup
Each project contains its own requirements.txt file with the necessary dependencies. To install the required libraries for any project, navigate to the project directory and run:

Setup
Each project contains its own requirements.txt file with the necessary dependencies. To install the required libraries for any project, navigate to the project directory and run:

pip install -r requirements.txt

Environment Setup
API Keys: Set up environment variables for projects that require API access (such as OpenAI and Tasty API) by creating a config.json file or setting environment variables as per project instructions.
Future Enhancements
Each project offers opportunities for further expansion:

Enhanced Clustering and Similarity Analysis: Add document clustering in NLP Document Enrichment for detailed similarity analysis.
Multi-document Summarization: Extend the Semantic Search Chatbot with cross-document summarization.
Machine Learning Classification: Introduce machine learning in Report Text Categorization for enhanced topic classification.
Enhanced Meal Suggestion Options: Add filters for health-related options or ingredient availability in the Recipe Generator App.
Contributing
Contributions are welcome! Follow these steps:

Fork the repository.
Create a feature branch (git checkout -b feature/YourFeature).
Commit your changes (git commit -m 'Add your message here').
Push to the branch (git push origin feature/YourFeature).
Open a pull request.


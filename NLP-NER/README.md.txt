Project Overview
This project automates the categorization of report text data by analyzing keywords related to three main topics:

Politics and War Conflict
Human Rights and Social Issues
Economic Impact and Infrastructure
The script reads the input data, tokenizes and cleans it, checks for specific keywords, and assigns categories accordingly. It then saves the categorized data as a new Excel file.

Features
Text Processing: Cleans and tokenizes text data, removing punctuation and stopwords for better categorization.
Keyword-based Categorization: Uses predefined lists of keywords to classify text into specific topics.
Automated Workflow: Automates the reading, processing, and categorization of text data, saving results in Excel format for easy review.

Setup Instructions
Clone the Repository:

git clone https://github.com/yourusername/Report-Text-Categorization.git
cd Report-Text-Categorization

Customization Options
You can modify the categorization by updating the keyword lists in the script:

war_related_terms: Keywords for political and military conflicts.
human_rights_related_terms: Keywords for human rights and social issues.
economy_related_terms: Keywords related to economic impact and infrastructure.
If you want to categorize reports into additional topics, simply create a new list of keywords and add another category check in the categorize_text function.

Future Enhancements
Consider adding the following enhancements for more robust categorization:

Machine Learning Classifier: Replace keyword-based categorization with an ML model (e.g., logistic regression or a simple neural network) for more accurate classification.
Named Entity Recognition (NER): Use NER to identify specific entities (like organizations or locations) for a finer-grained analysis.
Sentiment Analysis: Analyze the tone of each report (e.g., positive, neutral, negative) to understand the context better.
Visualization: Create charts or graphs summarizing the distribution of categories or showing trends over time.
Dependencies
pandas: For loading, processing, and saving Excel files.
nltk: For text tokenization, stopword removal, and preprocessing.
openpyxl: For handling Excel files.
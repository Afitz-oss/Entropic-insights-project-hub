import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# download the stopwords from NLTK
nltk.download('stopwords')

war_related_terms = ['military', 'attack', 'battle', 'conflict', 'war', 'ukrainian', 'russian', 'forces', 'troops', 'army', 'warfare', 'combat', 'soldier', 'weapon', 'missile', 'strike', 'raid', 'aircraft', 'tank', 'navy', 'marine', 'offensive', 'defensive', 'firefight', 'skirmish', 'ambush', 'military losses', 'military operation', 'military leadership', 'military intelligence']
human_rights_related_terms = ['rights', 'child', 'human', 'social', 'civillian', 'justice', 'liberty', 'equality', 'abuse', 'discrimination', 'exploitation', 'oppression', 'protest', 'refugee', 'asylum', 'displaced', 'homeless', 'victim', 'violence', 'crime', 'law', 'police', 'prison', 'punishment', 'corruption', 'civillian impact']
economy_related_terms = ['fuel', 'power', 'energy', 'infrastructure', 'pmc', 'economy', 'trade', 'import', 'export', 'market', 'stock', 'investment', 'currency', 'inflation', 'debt', 'deficit', 'unemployment', 'tax', 'tariff', 'recession', 'finance', 'bank', 'loan', 'credit', 'industry', 'manufacture', 'agriculture', 'transportation', 'logistics', 'infrastructure impact']


# load the dataset
df = pd.read_excel(r'C:\Users\AkimFitzgerald\Downloads\All Reports Master Database (1).xlsx')

stop_words = set(stopwords.words('english'))

# Function to clean and tokenize the 'Topic Content' column
def clean_tokenize(row):
    tokens = word_tokenize(row.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

df['Tokenized Content'] = df['Topic Content Data'].apply(clean_tokenize)

# Functions to assign categories based on keywords
def check_keywords(row, keyword_list):
    return any(word in row for word in keyword_list)

def categorize_text(row):
    categories = []
    if check_keywords(row, war_related_terms):
        categories.append('Politics and War Conflict')
    if check_keywords(row, human_rights_related_terms):
        categories.append('Human Rights and Social Issues')
    if check_keywords(row, economy_related_terms):
        categories.append('Economic Impact and Infrastructure')
    
    return categories if categories else ['Uncategorized']

df['Categories'] = df['Tokenized Content'].apply(categorize_text)
df.drop(columns=['Tokenized Content'], inplace=True)  # Drop the tokenized column

# Save the dataframe
df.to_excel(r'C:\Users\AkimFitzgerald\Downloads\All Reports Master Database (1)_new.xlsx')


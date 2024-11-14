import pandas as pd
import numpy as np
import textdistance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Read the Excel file containing the name matching dataset
df = pd.read_excel(r"c:\Users\btofi\Entropic-insights-project-hub\Sponsor name check algorithm - GR\data\name_matching_dataset.xlsx")


def extract_features(df):
    # Calculate Jaro-Winkler similarity for each name pair
    df['Jaro-Winkler'] = df.apply(lambda row: textdistance.jaro_winkler(row['Name Number 1'], row['Name Number 2']), axis=1)
    # Calculate Levenshtein similarity for each name pair
    df['Levenshtein'] = df.apply(lambda row: 1 - textdistance.levenshtein.normalized_distance(row['Name Number 1'], row['Name Number 2']), axis=1)
    return df

df = extract_features(df)

# Independant features
X = df[['Jaro-Winkler', 'Levenshtein']]
# Dependent feature
y = df['Match']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Evaluate the model's performance using accuracy, precision, recall, and F1 score
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')


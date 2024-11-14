Recipe Generator and AI Meal Suggestion App
Overview
This is a web-based application built with Dash that provides personalized meal suggestions using the OpenAI API and retrieves recipes from the Tasty API. The application allows users to input their preferences, search for recipes, and receive meal suggestions that align with their dietary and ingredient preferences.

Installation
Clone the repository and navigate to the project folder.
Install required dependencies:
pip install -r requirements.txt

Add your OpenAI and Tasty API keys as environment variables:
export OPENAI_API_KEY='your_openai_key'
export TASTY_API_KEY='your_tasty_api_key'

Usage
Start the App:
python <filename>.py
Open the App:

Navigate to http://127.0.0.1:8050/ in your web browser.
Interface:

Enter your name, select preferred cuisines, and hit "Get AI Meal Suggestion" for personalized meal ideas.
Alternatively, search for a recipe by entering keywords, ingredients, or setting a minimum rating.




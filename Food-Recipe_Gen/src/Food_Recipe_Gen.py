import pandas as pd
from openai import OpenAI
import requests
import random
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import os
client = OpenAI()

# Set up OpenAI API key from environment variable
client.api_key = os.getenv("OPENAI_API_KEY")

TASTY_API_KEY = os.getenv("TASTY_API_KEY")

# Load the food database
def load_food_database(file_path):
    df = pd.read_excel(file_path)
    return df.to_dict(orient='records')

food_db = load_food_database(r"c:\Users\btofi\OneDrive\Documents\Akim's Food Memory .xlsx")

# Get all unique cuisines from the database
def get_all_cuisines(food_db):
    cuisines = set()
    for entry in food_db:
        cuisines.update(entry["Preferred Cuisine List"].split(", "))
    return sorted(list(cuisines))

all_cuisines = get_all_cuisines(food_db)

# ChatGPT API integration
def get_chatgpt_response(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that suggests meals based on user preferences."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def get_ai_meal_suggestion(user_name, selected_cuisines):
    user_prefs = next((item for item in food_db if item["User Name"] == user_name), None)
    
    if not user_prefs:
        return "User not found in the database."

    prompt = f"Suggest a meal for {user_name} considering the following parameters:\n\n"
    prompt += f"1. Selected cuisines: {', '.join(selected_cuisines)}\n"
    prompt += f"2. Dietary restrictions: {user_prefs['Dietary Restrictions']}\n"
    prompt += f"3. Favorite ingredients: {user_prefs['Favorite Ingredients']}\n"
    prompt += f"4. Other preferences: {user_prefs['Other Preferences']}\n\n"
    prompt += "Please provide a meal suggestion that:\n"
    prompt += "- Comes from one of the selected cuisines\n"
    prompt += "- Respects the dietary restrictions\n"
    prompt += "- Incorporates favorite ingredients when possible\n"
    prompt += "- Takes into account other preferences\n"
    prompt += "Provide a brief description of the suggested meal and explain why it fits the user's preferences."

    return get_chatgpt_response(prompt)

def get_recipe_from_api(query, ingredients=None, min_rating=None):
    api_url = "https://tasty.p.rapidapi.com/recipes/list"
    headers = {
        "x-rapidapi-host": "tasty.p.rapidapi.com",
        "x-rapidapi-key": TASTY_API_KEY
    }
    params = {"q": query, "from": "0", "size": "20"}
    
    if ingredients:
        params["ingredients"] = ingredients
    if min_rating:
        params["min_rating"] = min_rating

    response = requests.get(api_url, headers=headers, params=params)
    return response.json()

def format_recipe(recipe):
    title = recipe.get('name', 'No title available')
    instructions = "\n".join([f"{i+1}. {step['display_text']}" for i, step in enumerate(recipe.get('instructions', []))])
    return f"Recipe: {title}\n\nInstructions:\n{instructions}"

app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    html.H1('Recipe Generator'),
    dcc.Input(id='user-input', type='text', placeholder='Enter your name'),
    html.Div([
        html.H3('Select Cuisines:'),
        dcc.Checklist(
            id='cuisine-checklist',
            options=[{'label': cuisine, 'value': cuisine} for cuisine in all_cuisines],
            value=all_cuisines,  # All cuisines selected by default
            inline=True
        )
    ]),
    html.Button('Get AI Meal Suggestion', id='ai-suggestion-button'),
    html.Div(id='ai-suggestion-output'),
    html.Hr(),
    dcc.Input(id='manual-query', type='text', placeholder='Enter your recipe search query'),
    dcc.Input(id='ingredient-input', type='text', placeholder='Enter ingredients (comma-separated)'),
    dcc.Input(id='min-rating-input', type='number', placeholder='Minimum user rating', min=0, max=5, step=0.1),
    html.Button('Search Recipe', id='search-button'),
    html.Div(id='recipe-output'),
    dcc.Store(id='session', storage_type='session')
])

@app.callback(
    Output('ai-suggestion-output', 'children'),
    Input('ai-suggestion-button', 'n_clicks'),
    State('user-input', 'value'),
    State('cuisine-checklist', 'value')
)
def update_ai_suggestion(n_clicks, user_name, selected_cuisines):
    if n_clicks is None or not user_name or not selected_cuisines:
        return "Please enter your name and select at least one cuisine."
    return get_ai_meal_suggestion(user_name, selected_cuisines)

@app.callback(
    Output('recipe-output', 'children'),
    Input('search-button', 'n_clicks'),
    State('manual-query', 'value'),
    State('ingredient-input', 'value'),
    State('min-rating-input', 'value')
)
def search_recipe(n_clicks, query, ingredients, min_rating):
    if n_clicks is None or not query:
        return "Enter a search query and click the button to search for a recipe."
    
    try:
        recipe_result = get_recipe_from_api(query, ingredients, min_rating)
        
        if recipe_result['count'] > 0:
            recipe = random.choice(recipe_result['results'])
            return format_recipe(recipe)
        else:
            return f"No recipes found for the query: {query}"
    except Exception as e:
        return f"An error occurred while fetching the recipe: {str(e)}"

if __name__ == '__main__':
    app.run_server(debug=True)
import faker
import pandas as pd 
import numpy as np 
from faker import Faker 
import random 
import openpyxl
from openpyxl import workbook
import os
import re

fake = Faker()

# Extracts the fake Sponsor Names from the text files 
def extract_names_from_files(folder_path):
    file_names = os.listdir(folder_path)
    parent_names = []

    for file_name in file_names:
        if file_name.endswith('.txt'):
            match = re.match(r'(.+?)\s(.+?)-', file_name)
            if match:
                first_name, last_name = match.groups()
                parent_names.append((first_name, last_name))

    return parent_names

# Function used to create columns that contain fake data 
def generate_data(num_rows, parent_names, max_repeated_names=15, num_names_from_files=100):
    # Initialize variables
    data = []
    used_parent_names = {}  # Dictionary to keep track of used parent names and their data
    repeated_names_count = 0
    names_from_files_count = 0

    # Ensuring num_names_from_files does not exceed the number of unique names in the text files
    num_names_from_files = min(num_names_from_files, len(parent_names))
    # For Loop that iterates over rows and creates fake data for each row
    for _ in range(num_rows):
        id_number = fake.unique.random_number(digits=9, fix_len=True)
        last_name = fake.last_name()
        first_name = fake.first_name()
        dob = fake.date_between(start_date='-100y', end_date='today')
        country = fake.country()
        date = fake.date_between(start_date='-10y', end_date='today')
        gender = random.choice(['Male', 'Female', 'Other'])
        # Constraints that are set for the Sponsor names that are generated 
        while True:
            if names_from_files_count < num_names_from_files:
                parent_first_name, parent_last_name = random.choice(parent_names)
                names_from_files_count += 1
            # If all names from the files have been used, generate new names
            else:
                while True:
                    parent_first_name, parent_last_name = fake.first_name(), fake.last_name()
                    parent_key = f"{parent_first_name} {parent_last_name}"
                    if parent_key not in used_parent_names:
                        break
            # Check if the current parent name is repeated
            parent_key = f"{parent_first_name} {parent_last_name}"
            repeated = parent_key in used_parent_names and used_parent_names[parent_key]['count'] > 0
            # Break the loop if conditions are met
            if not repeated or (repeated and used_parent_names[parent_key]['count'] < max_repeated_names and names_from_files_count <= num_names_from_files):
                break
        # Increment names_from_files_count if not exceeded limit
        if names_from_files_count < num_names_from_files:
            names_from_files_count += 1
        # If parent name is repeated, use existing data
        if repeated:
            used_parent_names[parent_key]['count'] += 1
            repeated_names_count += 1
            parent_dob, parental_relationship, cell_number, address, yes_or_no, length_living_somewhere, city, state, email, school = used_parent_names[parent_key]['data']
        # If parent name is not repeated, generate new data
        else:
            parent_dob = fake.date_between(start_date='-80y', end_date='-20y')
            parental_relationship = random.choice(['Mother', 'Father', 'Guardian'])
            cell_number = fake.phone_number()
            address = fake.address()
            yes_or_no = random.choice(['Yes', 'No'])
            length_living_somewhere = random.randint(1, 50)
            city = fake.city()
            state = fake.state()
            email = fake.email()
            second_address = fake.address()
            school = fake.company()
            # Save the newly generated data if the repeated names count is within the specified limit
            if repeated_names_count < max_repeated_names:
                used_parent_names[parent_key] = {
                    'count': 1,
                    'data': (parent_dob, parental_relationship, cell_number, address, yes_or_no, length_living_somewhere, city, state, email, school)
                }

        row = (id_number, last_name, first_name, dob, country, date, gender, parent_first_name, parent_last_name, parent_dob, parental_relationship, cell_number, address, yes_or_no, length_living_somewhere, city, state, email, second_address, school)
        data.append(row)

    return data

def save_to_excel(data, filename):
    columns = [
        'UC Portal A#', 'Last Name', 'First Name', 'DOB', 'COB', 'Date of Discharge ', 'Gender',
        'Sponsor First Name', 'Sponsor Last Name', 'Sponsor DOB ', 'Sponsor Relationship to UAC',
        'Sponsor Phone Number', 'Last known address', 'Address Flag in Portal? Address Used Multiple Times?', 'Length of Residency at Known Address',
        'City', 'State', 'Sponsor Email Address', 'Secondary Address', 'UAM School Name '
    ]
    
    df = pd.DataFrame(data, columns=columns)
    
    # Save DataFrame to Excel
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='RandomData')

if __name__ == '__main__':
    num_rows = 350
    folder_path = r"C:\Users\AkimFitzgerald\ORR Bulk Automation V2 Sample Stress Test (Not Sensitive Data)\Folder for text files"
    filename = 'Stress_test_data_1.xlsx'
    
    parent_names = extract_names_from_files(folder_path)
    data = generate_data(num_rows, parent_names, max_repeated_names=10, num_names_from_files=100)
    save_to_excel(data, filename)




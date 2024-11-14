Name Extraction and Sponsor Data Generation
This script extracts sponsor names from text files and generates a comprehensive Excel dataset of children and their sponsors, using a mix of real names and generated data. The resulting dataset is suitable for stress testing and validating data pipelines.

Purpose
Designed for generating synthetic sponsor data, this script extracts names from existing text files, generates additional data fields, and consolidates the information into a structured Excel workbook. It can be used to simulate large datasets with realistic entries for system testing.

Functionality
Extracts Sponsor Names: Reads names from text files and incorporates them into the generated dataset.
Simulates Sponsor and Child Information: Uses Faker to generate realistic data for each field, such as addresses, dates of birth, phone numbers, and school names.
Excel Export: Compiles the data into a structured Excel workbook, ideal for testing data handling processes.
Key Functions
extract_names_from_files: Extracts names from the file names in a specified folder.
generate_data: Generates synthetic data for each row, using the extracted names where applicable.
save_to_excel: Saves the generated data into an Excel workbook with specified columns.
Usage
Set the path to the folder containing text files and define the number of rows.
Run the script to create an Excel workbook with the generated data.

Example:

python
Copy code

folder_path = "path/to/text/files"
filename = "Generated_Sponsor_Data.xlsx"
parent_names = extract_names_from_files(folder_path)
data = generate_data(350, parent_names)
save_to_excel(data, filename)

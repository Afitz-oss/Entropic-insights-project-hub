Automated Excel Update from Text Files
This script automates the extraction of specific fields from text files and updates corresponding Excel workbooks with the extracted data. Designed to streamline data entry and ensure consistency, this program helps reduce manual errors and speeds up data processing, particularly useful in handling large volumes of records, such as sponsor and child data for bulk report automation.

Purpose
This script was created to automate the process of updating Excel files with information extracted from text files. By automatically pulling fields like name, date of birth, address, and phone numbers from text files and placing them in structured Excel sheets, this solution replaces manual data entry, increasing efficiency and data accuracy.

Functionality
Extracts Data from Text Files: Searches for specified fields within each text file and retrieves information based on start and end markers for each field.
Updates Excel Sheets: Matches extracted text data with Excel sheet names or workbook names, inserting relevant information into designated cells.
Conditional Formatting: Highlights tabs in different colors based on data completeness:
Green: All required fields are filled.
Yellow: Partial data is filled.
Red: No matching data found.

Requirements
Python 3.8+
Libraries: pandas, openpyxl, tqdm

Install dependencies with:
pip install pandas openpyxl tqdm

Code Overview
1. Extract Keys from Text Files
def extract_keys(pattern, folder_path):

Purpose: Extracts keys (identifiers) from text file names and renames files for easier lookup.
Pattern Matching: Uses regular expressions to capture parts of the file name as keys.

2. Extract Fields from Text Files
def txt_to_dict(directory, fields_data_dict, keys):

Purpose: Reads text files and extracts specific fields based on provided start and end strings.
Dictionary Structure: Saves extracted fields in a dictionary, with file names (without extension) as keys for quick lookup.

3. Compile Data from Excel Workbooks
excel_data_frames = []
for workbook_name in workbook_names:

Purpose: Loads Excel workbooks into DataFrames for easier handling and manipulation.
Workbook Names: Each workbook name is stored as a reference, enabling efficient data processing.

4. Update Excel with Extracted Text Data
def update_excel_with_text_data(workbook_folder_path, text_file_folder_path, fields_data_dict):

def update_excel_with_text_data(workbook_folder_path, text_file_folder_path, fields_data_dict):

Purpose: Updates specific cells in Excel sheets with data extracted from text files.
Conditional Formatting: Applies bold font to headers, and colors tabs based on the completeness of data:
Green: All fields are populated.
Yellow: Some fields are missing data.
Red: No matching data found in the text files.
Usage
Set Folder Paths: Define the paths for text files (text_file_folder_path) and Excel workbooks (workbook_folder_path).
Run the Script: Execute the script to update Excel workbooks based on the data extracted from text files.

Example:
update_excel_with_text_data(workbook_folder_path, text_file_folder_path, fields_data_dict)
Field Mappings
The script uses fields_data_dict to define which sections to extract from text files and where to place them in the Excel sheets.

Reference_ID: Extracted between Reference ID: and Subject Information.
Sponsor_Name: Extracted between Name: and Date of Birth:.
Phone Number: Extracted between Possible Phones Associated with Subject: and Indicators.
Address: Extracted between Address Summary and Address Details.
DOB: Extracted between Date of Birth: and Gender:.
Employer: Extracted between Possible Employers and Address Summary.
Associates: Extracted between Possible Associates - Summary and Possible Associates - Details.
Error Handling
The script includes logging to record errors and processing information:

Warnings: If no match is found for a key, a warning is logged.
Errors: If any workbook fails to save, an error is logged.
Practical Applications
This script can be applied to scenarios where large numbers of records must be updated with data from external sources. It's particularly useful for:

Bulk Data Entry: Quickly updating fields in structured templates.
Automated Reporting: Populating predefined report formats with extracted data.
Data Validation: Ensuring that all records have complete information before final submission.




Bulk Sponsor and Children Workbook Generator
This program automates the creation of Excel workbooks for sponsors and unreported children, replacing a previously manual process. It produces structured workbooks with realistic synthetic data for each child, allowing data entry and processing workflows to be tested.

Purpose
The script is designed to streamline data entry for large datasets involving unreported children and their sponsors. By automating workbook generation, it enables faster data handling and consistent formatting, reducing manual workload.

Functionality
Workbook Creation: Generates Excel workbooks with realistic entries for unreported children and sponsors.
Custom Formatting: Applies cell styles, borders, and fills to simulate standard workbook templates.
Dynamic File Naming: Names sheets and files based on sponsor and child identifiers for easy identification.
Key Functions
create_workbook: Creates and styles an Excel workbook for each sponsor-child record.
process_row: Extracts and formats data from an Excel file into individual workbooks.
main loop: Loops through data entries, creating workbooks for each row in the input file.
Usage
Update the df variable with the path to your input Excel file.
Run the script to create workbooks for each record in the input file.

Example:
df = pd.read_excel('Stress_test_data.xlsx')
for index, row in df.iterrows():
    create_workbook(row, columns, index)

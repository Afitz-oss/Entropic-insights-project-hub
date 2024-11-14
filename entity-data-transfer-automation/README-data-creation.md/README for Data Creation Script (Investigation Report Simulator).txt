Investigation Report Data Simulator
This script generates simulated investigation report text files for missing children and their sponsors, mimicking real-world reports while preserving confidentiality. The generated files are intended for automated extraction and analysis, allowing users to test workflows on synthetic data.

Purpose
This script was developed to create synthetic data that resembles investigation reports. By simulating common data entry points such as names, dates of birth, addresses, and possible associates, it allows for testing data extraction methods while preserving privacy.

Functionality
Generates Realistic Personal Data: Using the Faker library, the script produces realistic data for fields like names, phone numbers, addresses, and associations.
Simulates Report Structure: Each file follows a standard format, mimicking actual investigation reports.
Automated File Creation: The script can generate a specified number of files with unique timestamp-based filenames for traceability.
Key Functions
random_dob: Generates a random date of birth within a specified year range.
random_gender: Randomly assigns gender.
create_txt_files: Writes the generated information to text files in a predefined format.
main: Loops through the specified number of files to create individual text files with unique identifiers.
Usage
Set the number of files to generate (num_files).
Run the script to create .txt files with randomized data.

Example:
num_files = 100
main(num_files)



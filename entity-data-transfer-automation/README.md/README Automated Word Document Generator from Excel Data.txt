Automated Word Document Generator from Excel Data
This script automates the generation of Word documents using data extracted from Excel workbooks. Designed to create a structured report for each record in the Excel files, this solution is ideal for generating summary reports that include personal identification details and sponsor information.

Purpose
This program was created to generate Word document reports based on structured data in Excel files. Each report includes key information about individuals and their sponsors, consolidating various data points into a cohesive, formatted Word document. The generated reports can be used for project summaries, identification verification, or case tracking.

Functionality
Extract Data from Excel: Loads Excel files and retrieves relevant fields, such as sponsor name, address, email, phone number, and other identifiers.
Generate Formatted Word Documents: Creates a Word document for each record, populating it with extracted data and applying custom formatting.
Custom Header and Footer: Adds a logo in the header and a confidential disclaimer in the footer for each report.
Table Insertion and Styling: Inserts tables to display information about sponsors and associates, and styles them for a professional presentation.
Requirements
Python 3.8+
Libraries: pandas, docx (Python-Docx), skimpy
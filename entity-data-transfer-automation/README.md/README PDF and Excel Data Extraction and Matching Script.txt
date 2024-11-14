PDF and Excel Data Extraction and Matching Script
This script automates the extraction of data from PDF files, filters for specific content, and updates matching Excel workbooks with relevant information. The script is particularly useful for processing large volumes of PDFs and storing specific data, like social media links, in corresponding Excel sheets.

Purpose
This program was created to parse and extract tables from PDF files, filter content for specific social media URLs, and match these results with existing Excel workbooks. When a PDF file name closely matches an Excel sheet name, the script creates a new "Social Media Data" sheet within the Excel workbook and saves the filtered data.

Functionality
Renaming PDF Files: Modifies PDF file names for consistency by removing text after a dash.
PDF Content Extraction: Reads each PDF and extracts table-like data by splitting rows and cells.
Social Media Filtering: Identifies and filters rows containing linkedin.com, twitter.com, or facebook.com URLs.
Excel Matching and Updating: Matches PDF files to corresponding Excel sheet names (using a similarity threshold) and writes filtered data into a new sheet called "Social Media Data" if a match is found.
Requirements
Python 3.8+
Libraries: pandas, PyPDF2, openpyxl, textdistance
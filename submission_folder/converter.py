import os
import zipfile
import argparse


def create_zip_with_renamed_csv(csv_file_path):
    # Check if the file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: The file '{csv_file_path}' does not exist.")
        return
    
    # Get the base name (without extension) and directory of the file
    base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    zip_file_name = f"{base_name}_text.zip"
    
    try:
        # Create a ZIP file
        with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add the CSV file to the ZIP, renaming it to 'submission.csv'
            zipf.write(csv_file_path, arcname="submission.csv")
        
        print(f"ZIP file '{zip_file_name}' created successfully with 'submission.csv' inside.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage

parser = argparse.ArgumentParser()

parser.add_argument('--filename', type=str, help='Filename of the csv')

args = parser.parse_args()

csv_file_path = args.filename  # Replace with your CSV file path
create_zip_with_renamed_csv(csv_file_path)

import pandas as pd
import argparse
import os
import json
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--label', type=str, help='Label of the category.', required=True)
parser.add_argument('-i', '--input_json', type=str, help='Path to a JSON file with lists of subfolders containing predictions.', required=True)
parser.add_argument('-n', '--num_files', type=int, help='Number of files to select from the list for the given label.', required=True)

args = parser.parse_args()

ensemble_dir = './scripts/LOGS/LOGS'  # Directory for ensemble predictions

if args.label in ['hazard-category', 'product-category']:
    submission_files = ['submission_st1.csv', 'test_submission_st1.csv']
elif args.label in ['hazard', 'product']:
    submission_files = ['submission_st2.csv', 'test_submission_st2.csv']
else:
    raise ValueError("Invalid label.")

with open(args.input_json, 'r') as f:
    input_data = json.load(f)

if args.label not in input_data:
    raise ValueError(f"Label '{args.label}' not found in the input JSON.")

input_folders = input_data[args.label]
selected_folders = input_folders[:args.num_files]


def ensemble_predictions(label, files):
    votes = pd.concat([file[label] for file in files], axis='columns')
    return votes.mode(axis='columns').iloc[:, 0], len(files)


def process_files(label, selected_folders, submission_file, output_dir):
    files = []
    for input_folder in selected_folders:
        file_path = os.path.join(ensemble_dir, input_folder, submission_file)
        files.append(pd.read_csv(file_path))

    ensembled, number_of_files = ensemble_predictions(label, files)

    if 'test' in submission_file:
        output_file_name = f'ensemble_{label}_test.csv'
    else:
        output_file_name = f'ensemble_{label}.csv'

    output_csv_path = os.path.join(output_dir, output_file_name)
    pd.DataFrame({label: ensembled}).to_csv(output_csv_path, index=False)

    return output_file_name


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
base_output_dir = 'ensembled_results'
os.makedirs(base_output_dir, exist_ok=True)
output_dir = os.path.join(base_output_dir, f'ens_{len(selected_folders)}_{args.label}_{timestamp}')
os.makedirs(output_dir, exist_ok=True)

all_input_files = set()

output_file_names = []
for submission_file in submission_files:
    output_file_name = process_files(args.label, selected_folders, submission_file, output_dir)
    output_file_names.append(output_file_name)

output_txt_path = os.path.join(output_dir, 'input_files.txt')
with open(output_txt_path, 'w') as f:
    for file_path in selected_folders:
        f.write(file_path + '\n')

print(f"Ensembling completed for {args.label}. Results saved in {output_dir} with files: {', '.join(output_file_names)}.")

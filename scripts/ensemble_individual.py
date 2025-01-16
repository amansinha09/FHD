import pandas as pd
import argparse
import os
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--label', type=str, help='Label of the category.')
parser.add_argument('-i1', '--input1', type=str, help='Subfolder 1 with predictions.')
parser.add_argument('-i2', '--input2', type=str, help='Subfolder 2 with predictions.')
parser.add_argument('-i3', '--input3', type=str, help='Subfolder 3 with predictions.')
parser.add_argument('-i4', '--input4', type=str, help='Subfolder 4 with predictions.')
parser.add_argument('-i5', '--input5', type=str, help='Subfolder 5 with predictions.')
args = parser.parse_args()

ensemble_dir = './scripts/LOGS/LOGS'  # inside FHD

if args.label in ['hazard-category', 'product-category']:
    submission_file = 'submission_st1.csv'
elif args.label in ['hazard', 'product']:
    submission_file = 'submission_st2.csv'
else:
    raise ValueError("Invalid label.")

# Load prediction files from the subfolders
file1 = pd.read_csv(f'{ensemble_dir}/{args.input1}/{submission_file}')
file2 = pd.read_csv(f'{ensemble_dir}/{args.input2}/{submission_file}')
file3 = pd.read_csv(f'{ensemble_dir}/{args.input3}/{submission_file}')
file4 = pd.read_csv(f'{ensemble_dir}/{args.input4}/{submission_file}')
file5 = pd.read_csv(f'{ensemble_dir}/{args.input5}/{submission_file}')

def ensemble_predictions(label, files):
    votes = pd.concat([file[label] for file in files], axis='columns')
    return votes.mode(axis='columns').iloc[:, 0]

ensembled = ensemble_predictions(args.label, [file1, file2, file3, file4, file5])

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

base_output_dir = 'ensembled_results'
os.makedirs(base_output_dir, exist_ok=True)

output_dir = os.path.join(base_output_dir, f'ensemble_{args.label}_{timestamp}')
os.makedirs(output_dir, exist_ok=True)

output_csv_path = os.path.join(output_dir, f'ensemble_{args.label}.csv')
pd.DataFrame({args.label: ensembled}).to_csv(output_csv_path, index=False)

# Create a text file listing the input files used
input_files_list = [
    f'{ensemble_dir}/{args.input1}/{submission_file}',
    f'{ensemble_dir}/{args.input2}/{submission_file}',
    f'{ensemble_dir}/{args.input3}/{submission_file}',
    f'{ensemble_dir}/{args.input4}/{submission_file}',
    f'{ensemble_dir}/{args.input5}/{submission_file}'
]

output_txt_path = os.path.join(output_dir, 'input_files.txt')
with open(output_txt_path, 'w') as f:
    for file_path in input_files_list:
        f.write(file_path + '\n')

print(f"Sub-task category {args.label} ensembling completed and saved in {output_dir}.")

# # Merge the four ensembled outputs into a final submission file
# final_submission = pd.DataFrame({
#     'hazard-category': hazard_category,
#     'product-category': product_category,
#     'hazard': hazard,
#     'product': product
# })
#
# final_submission.to_csv('final_ensembled_submission.csv', index=False)
# print("Final submission file created!")

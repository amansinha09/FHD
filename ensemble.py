import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-input1', '--input1', type=str, help='Subfolder 1 with predictions.')
parser.add_argument('-input2', '--input2', type=str, help='Subfolder 2 with predictions.')
parser.add_argument('-input3', '--input3', type=str, help='Subfolder 3 with predictions.')

args = parser.parse_args()

ensemble_dir = '/home/federica/food_detection/my_experiments/ensembling'

file1_st1 = pd.read_csv(f'{ensemble_dir}/{args.input1}/submission1.csv')
file1_st2 = pd.read_csv(f'{ensemble_dir}/{args.input1}/submission2.csv')
file2_st1 = pd.read_csv(f'{ensemble_dir}/{args.input2}/submission1.csv')
file2_st2 = pd.read_csv(f'{ensemble_dir}/{args.input2}/submission2.csv')
file3_st1 = pd.read_csv(f'{ensemble_dir}/{args.input3}/submission1.csv')
file3_st2 = pd.read_csv(f'{ensemble_dir}/{args.input3}/submission2.csv')

result_1 = {}
for label in ['hazard-category', 'product-category']:
    all_data = [file1_st1[label], file2_st1[label], file3_st1[label]]
    votes = pd.concat(all_data, axis='columns')
    result_1[label] = votes.mode(axis='columns').iloc[:, 0]

ensembled_1 = pd.DataFrame({'hazard-category': result_1['hazard-category'], 'product-category': result_1['product-category']})
# for now the file is saved in the current directory
ensembled_1.to_csv('ensembled_submission_st1.csv', index=False)
print("submission ST1 created!")

result_2 = {}
for label in ['hazard', 'product']:
    all_data = [file1_st2[label], file2_st2[label], file3_st2[label]]
    votes = pd.concat(all_data, axis='columns')
    result_2[label] = votes.mode(axis='columns').iloc[:, 0]

ensembled_2 = pd.DataFrame({'hazard': result_2['hazard'], 'product': result_2['product']})
# for now the file is saved in the current directory
ensembled_2.to_csv('ensembled_submission_st2.csv', index=False)
print("submission ST2 created!")

import pandas as pd
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument(
    '--files',
    type=str,
    nargs=4,
    help='Paths to the ensemble folders for hazard-category, product-category, hazard, and product. Must contain exactly one path each for hazard-category, product-category, hazard, and product.'
)
parser.add_argument(
    '--plm',
    type=str,
    help='Language model used.',
    default='bert'
)
args = parser.parse_args()

# Define the required categories
required_categories = {'hazard-category', 'product-category', 'hazard', 'product'}
files_dict = {}
provided_categories = set()

# Validate the files and ensure all required categories are included once
for file_path in args.files:
    matched_category = None
    for category in required_categories:
        if category in file_path.split('_'):
            if matched_category is not None:
                print(f"Error: File '{file_path}' matches multiple categories, which is ambiguous.")
                sys.exit(1)
            matched_category = category

    if matched_category is None:
        print(f"Error: File path '{file_path}' does not match any required categories.")
        sys.exit(1)

    if matched_category in provided_categories:
        print(f"Error: Duplicate category '{matched_category}' in file paths.")
        sys.exit(1)

    provided_categories.add(matched_category)
    files_dict[matched_category] = file_path

# Ensure all required categories are present
if provided_categories != required_categories:
    missing_categories = required_categories - provided_categories
    print(f"Error: Missing categories: {', '.join(missing_categories)}")
    sys.exit(1)

# Extract the common identifier for the output file names
number = '_'.join(files_dict['product'].split('_')[:2])

hazard_category_df = pd.read_csv(f'./ensembled_results/{files_dict["hazard-category"]}/ensemble_hazard-category.csv')
product_category_df = pd.read_csv(f'./ensembled_results/{files_dict["product-category"]}/ensemble_product-category.csv')
hazard_df = pd.read_csv(f'./ensembled_results/{files_dict["hazard"]}/ensemble_hazard.csv')
product_df = pd.read_csv(f'./ensembled_results/{files_dict["product"]}/ensemble_product.csv')

final_submission_st1 = pd.concat([
    hazard_category_df.rename(columns={'hazard_category': 'hazard-category'}),
    product_category_df.rename(columns={'product_category': 'product-category'}),
], axis=1)

final_submission_st2 = pd.concat([
    hazard_df.rename(columns={'hazard': 'hazard'}),
    product_df.rename(columns={'product': 'product'})
], axis=1)

final_submission_st1.to_csv(f'{number}_{args.plm}_submission_st1.csv', index=False)
print(f"Final submission file for ST1 created!")

final_submission_st2.to_csv(f'{number}_{args.plm}_submission_st2.csv', index=False)
print(f"Final submission file for ST2 created!")

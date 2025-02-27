# Food Hazard Detection
This repository contains the code for the paper "Fossils at SemEval-2025 Task 9: Tasting Loss Functions for Food
Hazard Detection in Text Reports" (forth.).


## How to run

1. To obtain predictions: 
```
python scripts/main_loss.py --data_dir <data_dir> --which_loss <loss_function> --epochs <num_epochs> --input <title|text|title+text>
```

2. To run the evaluation script:
```
cd scripts/
python run_eval.py --gold ../data/incidents_valid.csv --pred <path-to-submission-file>
```

3. To run ensembling on individual categories:
```
python scripts/ensemble_individual.py -l <category (e.g. hazard)> -i <inputs> -n <num_files>
```

where `inputs` is the path to a JSON file listing subfolders with predictions files, and
`num_files` is the number of files to select from the list in `inputs` for the given label.

4. To merge ensembled results into a single submission file:
```
python merge_ensembles.py --hazard_category path/to/ensemble_hazard_category.csv  --product_category path/to/ensemble_product_category.csv --hazard path/to/ensemble_hazard.csv --product path/to/ensemble_product.csv
```

5. To convert the prediction file to the file format required for submission:
```
python submission_folder/converter.py --filename <name_of_csv_file.csv>
```

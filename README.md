# FHD
Code repository for project Food hazard detection Challenge


## To convert formatted file to submission zip required file format
```
python submission_folder/converter.py --filename <name_of_csv_file.csv>
```

## To run eval script
```
cd scripts/
python run_eval.py --gold ../data/incidents_valid.csv --pred <path-to-submission-file>
```

# scaling-disco
State-of-the-art relation extraction

Our sample data is SemEval 2010 TASK 8, which is available at [http://semeval2.fbk.eu/semeval2.php?location=data](http://semeval2.fbk.eu/semeval2.php?location=data)

We are using Python 2.7 for this project. Install dependent packages with 

`pip install -r requirements.txt`

Within relation_extraction folder, train and test a relation classifier with command 

`python relationExtraction.py TRAIN_FILE_NAME TEST_FILE_NAME -o TEST_REPORT_FILE_NAME`

If TEST_REPORT_FILE_NAME is not specified, the report will be printed to stdout.

Make sure your training data contains all relations that show up in testing data.

Run `python relationExtraction.py -h` for more information about command options.

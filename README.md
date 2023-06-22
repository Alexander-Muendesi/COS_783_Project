# Project Description
This project aimed to created a searching tool that would help forensic examiners be able to search efficiently through data.
# Project Setup Instructions

This project makes use of a virtual environment.

To activate the virtual environment, ensure you are in the `COS_783_Project` directory.

## Windows

To activate the virtual environment on Windows, type the following command in the command prompt:
```shell
.\KeywordSearcher\Scripts\activate
```

## MacOS or Linux

To activate the virtual environment on MacOS or Linux, use the following command in the terminal:
```shell
source KeywordSearcher/bin/activate
```

To install necessary packages, in the `COS_783_Project` directory, run the following command:

```shell
pip install -r requirements.txt
```

Additionally run the following command: 
```shell
python -m spacy download en_core_web_sm
```
When running project for first time make sure to uncomment line 3 in textPreprocessor.py so that it downloads wordnet. Then
comment it out as you won't need to download it again on future runs

Please follow these instructions to set up and configure your project environment.

To deactivate your virtual environment use the command "deactivate"

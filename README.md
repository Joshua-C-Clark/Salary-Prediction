# Salary Predictions
<br>

## Overview
This project predicts the salary for various tech industry roles based on:
- Potential Employer
- Job Location
- Title
- Job Subspecialty/Department
- Role

## Environment and Tools
1. scikit-learn
2. pandas
3. numpy
4. flask

## Installation

`pip install sklearn pandas numpy flask`

`python app.py`

## Data
Data for the project was obtained via web scraping scripts that were developed using Selenium. The scripts extracted data from dynamically loaded Javascript tables. For an overview of how to scrape a dynamically loaded table, please view my post on [Web Scraping](https://medium.com/@jcclark141152/data-extraction-from-dynamic-tables-9d9eafbd8064). Preliminary data analysis and preparation were completed in Jupyter Notebooks external to this project. I would be happy to share portions of these notebooks for any interested individuals.

## Model Training and Selection
The model was trained using a Random Forest Regression Ensemble based upon the inherent group "binning" of the feature columns into a series of discrete variables. 

The model is currently performing at an R^2 of 0.5. This should be improved upon with some of the items detailed below in future work.

## Future Work
- Incorporate re to improve data cleaning.
- Better user input processing to match training data format and values.
- Prediction intervals
- app.py styling and formating.

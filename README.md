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
Data for the project was obtained via web scraping scripts that were developed using Selenium. The scripts extracted data from dynamically loaded Javascript tables. For an overview of how to scrape a dynamically loaded table, please view my post on [Web Scraping](https://medium.com/@jcclark141152/data-extraction-from-dynamic-tables-9d9eafbd8064). A sample scraping file has been provided (tpm.py). The largest hurdles for acquiring data were dealing with dynamically generated JavaScript tables and the website responsiveness. Particularly modal pop-up windows that would create timeouts for the web scraper.

## Notebooks
Preliminary data analysis and preparation were completed in Jupyter Notebooks. Please begin with preparation.ipynb for a detailed look at the data cleaning requirements. The primary issues revolved around extreme outliers, and large deviations in feature column inputs based on spacing and capitalization differences. Outliers were addressed by filtereing results > Q3 + 1.5 * IQR. Next, Categorical Features were transformed with str modules such as strip() and split(). Finally each category was evaluated for a threshold number feature values to include to encapsilate ~85% of data records. 

## Model Training and Selection
The model was trained using a Random Forest Regression Ensemble based upon the inherent group "binning" of the feature columns into a series of discrete variables. 

The model is currently performing at an R^2 of 0.63 for Test Set and R^2 of 0.70 for Training Set. 

## Future Work
- Better process user input from app.py. If there are no matches to existing variable, convert the input to 'Other'
- app.py styling and formating.
- Convert Total_Comp to Categorical Bins and evaluate Classification Models

# Salary Predictions
<br>

## Overview
This project predicts the salary for various tech industry roles based on:
- Potential Employer
- Job Location
- Title/Level
- Job Subspecialty/Department
- Role

## Environment and Tools
1. scikit-learn
2. pandas
3. numpy
4. flask

## Installation

`pip install sklearn pandas numpy flask`

`python train_model.py`

`python app.py`

## Data
Data for the project was obtained via web scraping scripts that were developed using Selenium. The scripts extracted data from dynamically loaded Javascript tables. For an overview of how to scrape a dynamically loaded table, please view my post on [Web Scraping](https://medium.com/@jcclark141152/data-extraction-from-dynamic-tables-9d9eafbd8064). A sample scraping file has been provided (tpm.py). The largest hurdles for acquiring data were dealing with dynamically generated JavaScript tables and the website responsiveness. Particularly modal pop-up windows that would create timeouts for the web scraper.

## Notebooks
Preliminary data analysis and preparation were completed in Jupyter Notebooks. Please begin with preparation.ipynb for a detailed look at the data cleaning requirements. The primary issues revolved around deviations in data recording. Specifically, there was a wide range of formatting for the Location feature. Next, the target variable (Total Compensation) required formatting to extract '$' and ',' before casting to an integer type. Outliers were addressed by filtereing the target variable results > Q3 + 1.5 * IQR. Finally, a threshold filter was applied to the categorical features. This was due to the high dimensionality and low value count of some of the feature variables. Production categorical features were encoded using OneHotEncoder. 

## Model Training and Selection
XGBoost Regression and Linear Regression, in addition to a handful of classification algorithms on binned Total_Comp, were evaluated for model depolyment. Ultimately, due to model simplicity and relative performance, Linear Regression was selected as the final model.

The model is currently performing at an R^2 of 0.60 for Test Set and R^2 of 0.62 for Training Set and RMSE of ~$55,000 for both Training and Test Sets. 

## Future Work
- Connect a database to the scraped data and develop a scheduled periodicity for scraping and updating the database. Retrain the model to validate performmance periodically.
- Apply CSS styling to app.py and index.html

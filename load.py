import os
import settings
import pandas as pd

def load():
    # Create a list of each dataset for iterating over to create a compsite dataframe
    files = os.listdir(settings.DATA_DIR)
    full_df = []

    # Iterate over each file, create a new column to track the roles for each csv, and append them into
    # the final/full dataframe 
    for f in files:
        data = pd.read_csv(os.path.join(settings.DATA_DIR, f), index_col= 0)
        data['Role'] = os.path.basename(f).split('.')[0].replace('_',' ').title()
        full_df.append(data)

    full_df = pd.concat(full_df)
    # Save the final dataframe for future cleaning and use for later
    full_df.to_csv(os.path.join(settings.PROCESSED_DIR, 'data.csv'))

load()

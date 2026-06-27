import pandas as pd
import os
from datetime import datetime 

# Define folder paths
temp_folder = "mon-velo/collecte_data/temp"
output_folder = "mon-velo/collecte_data/dispo_velos"
current_date = datetime.now()
output_file = os.path.join(output_folder, f'{current_date.strftime("%d-%m-%Y")}.csv')

print(f'Daily execution {current_date}')

dataframes = []

for file_name in os.listdir(temp_folder):
    if file_name.endswith(".csv"):
        file_path = os.path.join(temp_folder, file_name)
        df = pd.read_csv(file_path)
        dataframes.append(df)

# Concatenate all dataframes into one
if dataframes:  # Check whether there are any dataframes to concatenate
    combined_df = pd.concat(dataframes, ignore_index=True).sort_values(by='number')
    # Drop duplicates
    combined_df.drop_duplicates(inplace=True)
    # Save the deduplicated dataframe to the dispo_velos folder
    combined_df.to_csv(output_file, index=False)
    print(f"File {output_file} was created successfully with duplicates removed.")

    # Clear out the temp folder
    for file_name in os.listdir(temp_folder):
        file_path = os.path.join(temp_folder, file_name)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error while deleting file {file_path}: {e}")
else:
    print("No CSV file found in the temp folder.")

print('Daily processing was a success')

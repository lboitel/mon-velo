import pandas as pd
import os
from datetime import datetime 

# Définir les chemins des dossiers
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

# Concaténer tous les dataframes en un seul
if dataframes:  # Vérifier s'il y a des dataframes à concaténer
    combined_df = pd.concat(dataframes, ignore_index=True).sort_values(by='number')
    # Supprimer les doublons
    combined_df.drop_duplicates(inplace=True)
    # Sauvegarder le dataframe sans doublons dans le dossier dispo_velos
    combined_df.to_csv(output_file, index=False)
    print(f"Le fichier {output_file} a été créé avec succès sans doublons.")

    # Supprimer le contenu du dossier temp
    for file_name in os.listdir(temp_folder):
        file_path = os.path.join(temp_folder, file_name)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Erreur lors de la suppression du fichier {file_path}: {e}")
else:
    print("Aucun fichier CSV trouvé dans le dossier temp.")

print('Daily processing was a success')

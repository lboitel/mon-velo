import os 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import glob
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm 

current_date = datetime.now()
print(f'Daily execution {current_date}')


data_path = 'mon-velo/collecte_data/dispo_velos'
file_names = os.listdir(data_path)

data = pd.concat([pd.read_csv(f'{data_path}/{file_name}') for file_name in file_names])


data['total_bikes'] = data['available_bike_stands'] + data['available_bikes']
data['percentage_bikes_available'] = data.apply(
    lambda row: (row['available_bikes'] / row['total_bikes']) if row['total_bikes'] > 0 else 0,
    axis=1
)

data['timestamp'] = pd.to_datetime(data['last_update'], utc=True)
data['day_of_week'] = data['timestamp'].dt.strftime('%A')

day_of_week_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
data['day_of_week'] = data['day_of_week'].map(day_of_week_mapping)

data['hour_of_day'] = data['timestamp'].dt.hour

features = ['number', 'day_of_week', 'hour_of_day']
target = 'percentage_bikes_available'

X = data[features]
y = data[target]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

directory_path = 'mon-velo/collecte_data/data_stations'

csv_files = glob.glob(os.path.join(directory_path, '*.csv'))

if not csv_files:
    raise FileNotFoundError("Aucun fichier CSV trouvé dans le répertoire spécifié.")

latest_file = max(csv_files, key=os.path.getctime)
print(latest_file)

df = pd.read_csv(latest_file)
df = df[df['status'] == 'OPEN']

unique_numbers = df['station_numbers'].unique()

from datetime import datetime, timezone


current_utc_time = datetime.now(timezone.utc)
day_of_week = current_utc_time.isoweekday()

preds = []
hours = []
interpretation = []
station_numbers = []

for station_number in tqdm(unique_numbers):
    for hour in range(24):
        station_numbers.append(station_number)
        hours.append(hour)
        pred = model.predict([[station_number, day_of_week, hour]])
        preds.append(pred)
        if pred > 0.8:
            interpretation.append('very likely')
        elif pred > 0.4:
            interpretation.append('likely')
        else: 
            interpretation.append('not likely')



current_date = datetime.now()
current_date = current_date.strftime("%d-%m-%Y")

output = pd.DataFrame({'station_numbers': station_numbers, 'hours': hours, 'preds': preds, 'commentary': interpretation})
output.to_csv(f'mon-velo/prediction_data/data_preds/preds_{current_date}.csv')

print('Predictions effectuees')
print('---------------------------------')
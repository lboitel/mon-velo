import requests as rq
import pandas as pd
from datetime import datetime
from dateutil import parser

def is_day_same(date_str):
    parsed_date = parser.parse(date_str)
    current_date = datetime.utcnow()

    return (parsed_date.year == current_date.year and
                parsed_date.month == current_date.month and
                parsed_date.day == current_date.day)


keys_to_collect = ['number', 'bike_stands', 'available_bike_stands', 'available_bikes', 'last_update']

output = {'number': [], 
          'bike_stands': [],
          'available_bike_stands': [],
          'available_bikes': [],
          'last_update': []
          }

resp = rq.get(url= 'https://data.toulouse-metropole.fr/api/explore/v2.1/catalog/datasets/api-velo-toulouse-temps-reel/records?limit=100').json()['results']
for data in resp:
    if is_day_same(data.get('last_update')):
        for k in keys_to_collect:
            output[k].append(data.get(k))

current_date = datetime.now()

output_df = pd.DataFrame(output).sort_values(by='number')
output_df.to_csv(f'mon-velo/collecte_data/temp/{current_date.strftime("%d-%m-%Y_%H:%M")}.csv', index=False)

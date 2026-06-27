import requests as rq
import pandas as pd
from datetime import datetime

def generate_number_to_name():
    
    station_names = []
    station_addresses = []
    station_numbers = []
    station_pos = []
    station_capacities = []
    status = []

    offset = 0
    limit = 100
    url = f'https://data.toulouse-metropole.fr/api/explore/v2.1/catalog/datasets/api-velo-toulouse-temps-reel/records'
    total_count = rq.get(url=url).json()['total_count']
    
    while total_count > 0:
        if total_count < 100:
            limit = total_count%100
        
        url = f'https://data.toulouse-metropole.fr/api/explore/v2.1/catalog/datasets/api-velo-toulouse-temps-reel/records?limit={limit}&offset={offset}'
        content = rq.get(url=url).json()['results']

        for station_resp in content:
            station_names.append(station_resp.get('name')[8:])
            station_addresses.append(station_resp.get('address'))
            station_pos.append((station_resp.get('position')['lon'], station_resp.get('position')['lat']))
            station_capacities.append(station_resp.get('bike_stands'))
            station_numbers.append(station_resp.get('number'))
            status.append(station_resp.get('status'))
        total_count -= 100
        offset +=100

    data = {'station_numbers': station_numbers,
            'station_names': station_names,
            'station_addresses': station_addresses,
            'station_pos': station_pos,
            'station_capacities': station_capacities,
            'status': status
            }
    
    df = pd.DataFrame(data).sort_values(by='station_numbers')
    current_date = datetime.now()
    print(f'WE COLLECTED {current_date.strftime("%d-%m-%Y")}')
    df.to_csv(f'mon-velo/collecte_data/data_stations/{current_date.strftime("%d-%m-%Y")}.csv', index=False)
    
generate_number_to_name()
import flwr as fl 
import pandas as pd 
import os
from multiprocessing.pool import ThreadPool

df = pd.read_csv('data/df_dummy.csv', on_bad_lines="skip")
countries = df.federation_level.unique()
# num_labels = len(df['y'].unique())  # Number of unique labels in the 'y' column
def run_client(ids):
    def async_client(_id):
        print(_id)
        os.system(f'python src/multi_class_text_classifier/pytorch/dummy_example/client.py {_id} &')
    client_pool = ThreadPool(processes=len(ids))
    client_pool.map(async_client, ids)

run_client(countries)
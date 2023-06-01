import flwr as fl
import pandas as pd
import multiprocessing as mp
import os 

df = pd.read_csv('models/firstTry/df_dummy.csv', on_bad_lines="skip")
countries = df.federation_level.unique()

def run_client(country):
    cmd = f'python /Users/kiliansprenkamp/Desktop/code/refugee_supervised_text_classification/src/multi_class_text_classifier/pytorch/dummy_example/run_client.py {country}'
    os.system(cmd)

# Create a separate process for each country
processes = [mp.Process(target=run_client, args=(country,)) for country in countries]

# Start all processes
for process in processes:
    process.start()

# Wait for all processes to finish
for process in processes:
    process.join()

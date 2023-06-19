import flwr as fl
import torch
import numpy as np
from typing import List, Tuple, Union, Optional, Dict
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Parameters, Scalar
import grpc
from flwr.server.strategy.dpfedavg_fixed import DPFedAvgFixed
#from flwr.server.strategy.dpfedavg_adaptive import DPFedAvgAdaptive

# Set the maximum message size (e.g., 1 GB)
max_message_size = 1024 * 1024 * 1024
channel = grpc.insecure_channel('127.0.0.1:8080', options=[
    ('grpc.max_receive_message_length', max_message_size),
    ('grpc.max_send_message_length', max_message_size)
])


Metrics = Dict[str, float]

def agg_metrics_val(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Collect all the FL Client metrics and weight them
    print(metrics)
    losses = [n_examples * metric['loss'] for n_examples, metric in metrics]
    accuracies = [n_examples * metric['accuracy'] for n_examples, metric in metrics]
    precision = [n_examples * metric['precision'] for n_examples, metric in metrics]
    recall = [n_examples * metric['recall'] for n_examples, metric in metrics]
    f1_score = [n_examples * metric['f1_score'] for n_examples, metric in metrics]

    total_examples = sum([n_examples for n_examples, _ in metrics])

    # Compute weighted averages
    agg_metrics = {
        'val_loss': sum(losses) / total_examples
        , 'val_accuracy': sum(accuracies) / total_examples
        , 'val_precision': sum(precision) / total_examples
        , 'val_recall': sum(recall) / total_examples
        , 'val_f1_score': sum(f1_score) / total_examples
    }

    return agg_metrics

# def agg_metrics_train(metrics: List[Tuple[int, Metrics]]) -> Metrics:
#     # Collect all the FL Client metrics and weight them
#     losses = [n_examples * metric['loss'] for n_examples, metric in metrics]
#     accuracies = [n_examples * metric['accuracy'] for n_examples, metric in metrics]
#     precision = [n_examples * metric['precision'] for n_examples, metric in metrics]
#     recall = [n_examples * metric['recall'] for n_examples, metric in metrics]
#     f1_score = [n_examples * metric['f1_score'] for n_examples, metric in metrics]

#     total_examples = sum([n_examples for n_examples, _ in metrics])

#     # Compute weighted averages
#     agg_metrics = {
#         'train_loss': sum(losses) / total_examples
#         , 'train_accuracy': sum(accuracies) / total_examples
#         , 'train_precision': sum(precision) / total_examples
#         , 'train_recall': sum(recall) / total_examples
#         , 'train_f1_score': sum(f1_score) / total_examples
#     }

#     return agg_metrics

# Start Flower server

n_rounds = 10
# fl_strategy_FedAvg = fl.server.strategy.FedAvg(
#     # fit_metrics_aggregation_fn=agg_metrics_train, 
#     evaluate_metrics_aggregation_fn=agg_metrics_val)
fl_strategy_FedAvg = fl.server.strategy.FedAvg(fraction_fit=0.1,  # Sample 100% of available clients for training        
                                    # fraction_eval=1.0,  # Sample 100% of available clients for evaluation        
                                    min_fit_clients=2,  # Never sample less than 10 clients for training       
                                    min_evaluate_clients=2,  # Never sample less than 10 clients for evaluation        
                                    min_available_clients=2,  # Wait until all 10 clients are available
                                    # fit_metrics_aggregation_fn=agg_metrics_train, 
                                    evaluate_metrics_aggregation_fn=agg_metrics_val
                                    )
fl_strategy = DPFedAvgFixed(
    num_sampled_clients=2,
    noise_multiplier=1,
    clip_norm = 2,
    strategy=fl_strategy_FedAvg
    )
result = fl.server.start_server(server_address="127.0.0.1:8080", 
                                strategy=fl_strategy_FedAvg, 
                                config=fl.server.ServerConfig(num_rounds=n_rounds), 
                                grpc_max_message_length=1438900533)

# client_resources = None
# if DEVICE.type == "cuda":
#     client_resources = {"num_gpus": 1}

# # Start simulation
# fl.simulation.start_simulation(
#     client_fn=client_fn,
#     num_clients=NUM_CLIENTS,
#     config=fl.server.ServerConfig(num_rounds=5),
#     strategy=fl_strategy_FedAvg,
#     client_resources=client_resources,
# )

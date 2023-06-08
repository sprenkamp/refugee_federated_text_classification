import flwr as fl
import torch
import numpy as np
from typing import List, Tuple, Union, Optional, Dict
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Parameters, Scalar

Metrics = Dict[str, float]


def agg_metrics_val(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Collect all the FL Client metrics and weight them
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
n_rounds = 2
fl_strategy = fl.server.strategy.FedAvg(fit_metrics_aggregation_fn=agg_metrics_train, evaluate_metrics_aggregation_fn=agg_metrics_val)
result = fl.server.start_server(server_address="127.0.0.1:8080", strategy=fl_strategy, config=fl.server.ServerConfig(num_rounds=n_rounds))



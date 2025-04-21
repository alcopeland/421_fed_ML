"""insider-attack-scenario-pt: A Flower / PyTorch app."""

import torch
from torch.utils.data import DataLoader
from typing import List, Tuple
from datasets import load_dataset
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from insider_attack_scenario_pt.task import Net, get_weights, set_weights, test, get_transforms

def get_evaluate_fn(testloader, device):

    def evaluate(server_round, parameters_ndarray, config):
        net = Net()
        set_weights(net, parameters_ndarray)
        net.to(device)
        loss, accuracy = test(net, testloader, device)
        return loss, {"centralised_accuracy": accuracy}
    
    return evaluate


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """A function that aggregates metrics from clients."""
    accuracies = [num_samples * m["accuracy"] for num_samples, m in metrics]
    total_samples = sum(num_samples for num_samples, _ in metrics)
    return {"accuracy": sum(accuracies) / total_samples}

# Might be able to use this to simulate malicious clients, or use a malicious weighted average function
def on_fit_config(server_round: int) -> Metrics:
    """Adjusts the learning rate based on the server round."""
    lr = 0.01
    if server_round > 2:
        lr = 0.005
    return {"lr": lr}

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Load global test set
    testset = load_dataset("ylecun/mnist", split="test")
    testloader = DataLoader(testset.with_transform(get_transforms()), batch_size=32)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=on_fit_config,
        evaluate_fn = get_evaluate_fn(testloader, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)

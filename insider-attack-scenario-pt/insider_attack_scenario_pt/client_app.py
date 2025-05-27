"""insider-attack-scenario-pt: A Flower / PyTorch app."""

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from insider_attack_scenario_pt.task import Net, get_weights, load_data, set_weights, test, train

import random

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, partition_id, num_insiders, insiders, malicious_training_fn):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.partition_id = partition_id
        self.num_insiders = num_insiders
        self.malicious = partition_id in insiders
        self.malicious_training_fn = malicious_training_fn
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        
    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            config['lr'],
            self.device,
            self.malicious,
            self.malicious_training_fn,
        )
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy, precision, recall, f1 = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def client_fn(context: Context):
    # Load model and data
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    seed = context.run_config["seed"]
    trainloader, valloader = load_data(partition_id, num_partitions, seed)
    local_epochs = context.run_config["local-epochs"]
    insider_ratio = context.run_config["insider-ratio"]
    attack_type = context.run_config["attack-type"]
    num_insiders = int(num_partitions * insider_ratio)

    random.seed(seed)
    insiders = random.sample(range(1,num_partitions+1), num_insiders)

    malicious_training_fn = get_malicious_training_fn(attack_type)

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs, partition_id, num_insiders, insiders, malicious_training_fn).to_client()

def get_malicious_training_fn(attack_type):

    if attack_type == "general-label-flipping":
        def malicious_training_fn(net, trainloader, optimizer, criterion, device, running_loss, epochs):
            for _ in range(epochs):
                for batch in trainloader:
                    images = batch["image"]
                    labels = batch["label"]

                    labels = (labels + 1) % 10  # Flip labels

                    optimizer.zero_grad()
                    loss = criterion(net(images.to(device)), labels.to(device))
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

            avg_trainloss = running_loss / len(trainloader)
            return avg_trainloss

    elif attack_type == "targeted-label-flipping":
        # flip labels of a single class e.g. 1 -> 7
        def malicious_training_fn(net, trainloader, optimizer, criterion, device, running_loss, epochs):
            for _ in range(epochs):
                for batch in trainloader:
                    images = batch["image"]
                    labels = batch["label"]

                    labels = torch.where(labels == 1, 7, labels) # Flip labels

                    optimizer.zero_grad()
                    loss = criterion(net(images.to(device)), labels.to(device))
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

            avg_trainloss = running_loss / len(trainloader)
            return avg_trainloss

    elif attack_type == "general-model-poisoning":
        def malicious_training_fn(net, trainloader, optimizer, criterion, device, running_loss, epochs):
            for _ in range(epochs):
                for batch in trainloader:
                    images = batch["image"]
                    labels = batch["label"]

                    images += torch.randn_like(images) * 0.2  # Add noise to images
                    images = torch.clamp(images, 0, 1)  # Ensure images are in valid range

                    optimizer.zero_grad()
                    loss = criterion(net(images.to(device)), labels.to(device))
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

            avg_trainloss = running_loss / len(trainloader)
            return avg_trainloss

    elif attack_type == "targeted-model-poisoning":
        def malicious_training_fn(net, trainloader, optimizer, criterion, device, running_loss, epochs):
            for _ in range(epochs):
                for batch in trainloader:
                    images = batch["image"]
                    labels = batch["label"]

                    for i in range(images.shape[0]):
                        if labels[i] == 1:
                            images[i] += torch.randn_like(images[i]) * 0.7  # Add noise to images

                    optimizer.zero_grad()
                    loss = criterion(net(images.to(device)), labels.to(device))
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

            avg_trainloss = running_loss / len(trainloader)
            return avg_trainloss

    elif attack_type == "fixed-free-riding":
        def malicious_training_fn(net, trainloader, optimizer, criterion, device, running_loss, epochs):
            return 0.01
    
    elif attack_type == "random-free-riding":
        def malicious_training_fn(net, trainloader, optimizer, criterion, device, running_loss, epochs):
            with torch.no_grad():
                for name, param in net.named_parameters():
                    param.data = torch.randn_like(param.data) # * 0.1
            return 0.01

    elif attack_type == "backdoor":
        # modify a fraction of training images and change label to a specific label
        def malicious_training_fn(net, trainloader, optimizer, criterion, device, running_loss, epochs):
            for _ in range(epochs):
                for batch in trainloader:
                    images = batch["image"]
                    labels = batch["label"]

                    # Add a backdoor trigger to a fraction of the images
                    if random.random() < 0.7:
                        for i in range(images.shape[0]):
                            images[i, 0:5, 0:5] = 1.0
                        labels = torch.full_like(labels, 7)  # Change label to a specific class

                    optimizer.zero_grad()
                    loss = criterion(net(images.to(device)), labels.to(device))
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

            avg_trainloss = running_loss / len(trainloader)
            return avg_trainloss

    elif attack_type == "parameter-scaling":
        # multiply model parameters by a factor
        def malicious_training_fn(net, trainloader, optimizer, criterion, device, running_loss, epochs):
            for _ in range(epochs):
                for batch in trainloader:
                    images = batch["image"]
                    labels = batch["label"]
                    optimizer.zero_grad()
                    loss = criterion(net(images.to(device)), labels.to(device))
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

            for name, param in net.named_parameters():
                param.data *= 10  # Scale parameters

            avg_trainloss = running_loss / len(trainloader)
            return avg_trainloss
    
    else:
        malicious_training_fn = None

    return malicious_training_fn

# Flower ClientApp
app = ClientApp(
    client_fn,
)

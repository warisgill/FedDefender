import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision




class SimpleCNN(nn.Module):
    def __init__(self, config) -> None:
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(config["channels"], 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def initializeModel(config):
    net = None
    channels = config["channels"]
    num_classes = 10 
    if config["architecture"] == "simplecnn2":
        net = SimpleCNN(config)
    elif "resnet" in config["architecture"]:
        if "resnet18" == config["architecture"]:
            net = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        elif "resnet34" == config["architecture"]:
            net = torchvision.models.resnet34(weights="IMAGENET1K_V1")
        elif "resnet50" == config["architecture"]:
            net = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        elif "resnet101" == config["architecture"]:
            net = torchvision.models.resnet101(weights="IMAGENET1K_V1")
        elif "resnet152" == config["architecture"]:
            net = torchvision.models.resnet152(weights="IMAGENET1K_V1")

        if channels == 1:
            net.conv1 = torch.nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # set_parameter_requires_grad(net, feature_extract)
        num_ftrs = net.fc.in_features
        net.fc = torch.nn.Linear(num_ftrs, num_classes)
    
    elif config["architecture"] == "densenet121":
        """ Densenet
        """
        net = torchvision.models.densenet121(weights="IMAGENET1K_V1")
        if channels == 1:  
            net.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
        num_ftrs = net.classifier.in_features
        net.classifier = torch.nn.Linear(num_ftrs, num_classes)
    
    
    return net


def train(net, trainloader, epochs: int, DEVICE):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    nk_data_points = 0
    net = net.to(DEVICE)
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= total
        epoch_acc = correct / total
        nk_data_points = total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
    net = net.cpu()
    return nk_data_points 


def test(net, testloader, DEVICE):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    net =  net.to(DEVICE)
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    net = net.cpu()
    return loss, accuracy
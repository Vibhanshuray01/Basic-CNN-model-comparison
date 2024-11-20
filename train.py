import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
import random
import matplotlib.pyplot as plt
from model import MnistCNN

def train_model(kernel_numbers, model_name="model1", optimizer_name="adam", batch_size=32, epochs=1):
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Parameters
    BATCH_SIZE = batch_size
    EPOCHS = epochs
    LEARNING_RATE = 0.001

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model, loss function, and optimizer
    device = torch.device('cpu')
    model = MnistCNN(kernel_numbers).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Select optimizer based on parameter
    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    else:  # SGD
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # Training logs
    training_logs = {'loss': [], 'accuracy': [], 'model_name': model_name, 'kernels': kernel_numbers}

    def save_logs():
        with open(f'training_logs_{model_name}.json', 'w') as f:
            json.dump(training_logs, f)

    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            accuracy = 100 * correct / total

            # Save logs
            training_logs['loss'].append(loss.item())
            training_logs['accuracy'].append(accuracy)
            save_logs()

            if batch_idx % 100 == 0:
                print(f'{model_name} - Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')

    # Add test image generation after training
    def test_random_images(model, model_name):
        model.eval()
        test_data = datasets.MNIST('./data', train=False, transform=transform)
        
        # Select 10 random images
        indices = random.sample(range(len(test_data)), 10)
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()

        with torch.no_grad():
            for idx, i in enumerate(indices):
                image, label = test_data[i]
                output = model(image.unsqueeze(0))
                pred = output.argmax(dim=1, keepdim=True)

                axes[idx].imshow(image.squeeze(), cmap='gray')
                axes[idx].set_title(f'Pred: {pred.item()}\nTrue: {label}')
                axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(f'test_results_{model_name}.png')
        plt.close()

    # Generate test results after training
    test_random_images(model, model_name)
    print(f"\nTraining completed for {model_name}! Check test_results_{model_name}.png for model predictions on random test images.")
    
    return model, training_logs

if __name__ == '__main__':
    print("Training will start after you select kernel configurations in the web interface.") 
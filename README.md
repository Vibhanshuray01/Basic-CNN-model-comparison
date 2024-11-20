Interactive MNIST Training with Real-Time Visualization
An interactive Python project that trains a 4-layer Convolutional Neural Network (CNN) on the MNIST dataset, allowing users to customize training parameters, visualize real-time training logs and loss curves, and compare the performance of two different models. After training, the results are displayed on 10 randomly selected images from the MNIST dataset.

**Features**
Train a 4-Layer CNN: A simple yet effective CNN architecture for MNIST digit classification.
Real-Time Visualization: Training logs and loss curves are dynamically displayed on an HTML interface.
Interactive UI: Customize training parameters such as:
Optimizers (e.g., SGD, Adam)
Batch size
Number of kernels
Epoch count
Model Comparison: Evaluate and compare loss and accuracy of two different models.
Prediction Showcase: View predictions on 10 random images after training.

**Getting Started**

Prerequisites
Ensure you have the following installed:

Python 3.8 or higher
pip package manager
Required Libraries
Install dependencies using:


pip install -r requirements.txt

How It Works
Step 1: Launch the Application
Run the server using:


python app.py
The app will start and can be accessed at http://127.0.0.1:5000.

Step 2: Customize Training Parameters
On the web interface:

Select an optimizer (e.g., SGD or Adam).
Choose a batch size and the number of epochs.
Specify the number of kernels for convolutional layers.
Step 3: Train the Model
Click the Start Training button.
Watch the real-time training logs and loss curves update dynamically on the screen.
Step 4: Compare Models
Adjust parameters to train a second model.
The interface displays a side-by-side comparison of loss and accuracy for both models.
Step 5: View Predictions
Once training is complete:

The app picks 10 random MNIST images.
Displays predictions alongside the ground truth labels.

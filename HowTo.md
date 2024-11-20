# MNIST CNN with Live Training Visualization

This project implements a 4-layer CNN for MNIST digit classification with real-time training visualization and model comparison.

## Project Structure
project/
├── train.py          # Training logic
├── model.py          # CNN model definition
├── visualize.py      # Web server and visualization
└── templates/
    └── index.html    # Web interface

## Setup and Requirements

1. Install required packages:
bash
pip install torch torchvision flask numpy matplotlib

2. Start the visualization server:
bash
python visualize.py

3. Open your browser and go to:
http://127.0.0.1:5000/

4. In the web interface:
   - Enter kernel numbers for Model 1 (e.g., "16,32,64,64")
   - Enter kernel numbers for Model 2 (e.g., "8,16,32,32")
   - Click "Start Training"

5. Watch the training progress:
   - Blue line: Model 1 Loss
   - Green line: Model 1 Accuracy
   - Orange line: Model 2 Loss
   - Red line: Model 2 Accuracy

The training status and progress will be automatically updated in your browser.

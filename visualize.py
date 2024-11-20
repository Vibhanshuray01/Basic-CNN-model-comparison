from flask import Flask, render_template, jsonify, request
import json
from pathlib import Path
import threading
import time
import traceback
import sys

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Initialize with more detailed status
training_status = {
    'is_training': False,
    'current_model': None,
    'error': None,
    'last_update': None,
    'detailed_error': None
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def get_data():
    try:
        data = {'model1': None, 'model2': None}
        
        try:
            with open('training_logs_model1.json', 'r') as f:
                data['model1'] = json.load(f)
        except FileNotFoundError:
            pass
            
        try:
            with open('training_logs_model2.json', 'r') as f:
                data['model2'] = json.load(f)
        except FileNotFoundError:
            pass
            
        return jsonify(data)
    except Exception as e:
        print(f"Error in get_data: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)})

@app.route('/train', methods=['POST'])
def start_training():
    global training_status
    
    if training_status['is_training']:
        return jsonify({'status': 'error', 'message': 'Training already in progress'})
    
    try:
        data = request.json
        print(f"Received training request with data: {data}")  # Debug print
        
        kernels1 = [int(k) for k in data['kernels1']]
        kernels2 = [int(k) for k in data['kernels2']]
        optimizer = data['optimizer']
        batch_size = data['batch_size']
        epochs = data['epochs']
        
        def train_models():
            global training_status
            try:
                print("Starting training process...")  # Debug print
                training_status['is_training'] = True
                training_status['error'] = None
                training_status['detailed_error'] = None
                training_status['last_update'] = time.time()
                
                # Import here to avoid circular imports
                from train import train_model
                
                # Clear previous training logs
                for model in ['model1', 'model2']:
                    try:
                        Path(f'training_logs_{model}.json').unlink()
                    except FileNotFoundError:
                        pass
                
                print("Training Model 1...")  # Debug print
                training_status['current_model'] = 'model1'
                training_status['last_update'] = time.time()
                train_model(kernels1, 'model1', optimizer, batch_size, epochs)
                
                print("Training Model 2...")  # Debug print
                training_status['current_model'] = 'model2'
                training_status['last_update'] = time.time()
                train_model(kernels2, 'model2', optimizer, batch_size, epochs)
                
                print("Training completed successfully")  # Debug print
                
            except Exception as e:
                print(f"Error during training: {str(e)}")  # Debug print
                traceback.print_exc()
                training_status['error'] = str(e)
                training_status['detailed_error'] = traceback.format_exc()
            finally:
                training_status['is_training'] = False
                training_status['current_model'] = None
                training_status['last_update'] = time.time()
        
        # Start training in a separate thread
        thread = threading.Thread(target=train_models)
        thread.daemon = True  # Make thread daemon so it dies with the main process
        thread.start()
        
        return jsonify({'status': 'success', 'message': 'Training started'})
        
    except Exception as e:
        print(f"Error in start_training: {str(e)}")  # Debug print
        traceback.print_exc()
        training_status['error'] = str(e)
        training_status['detailed_error'] = traceback.format_exc()
        training_status['is_training'] = False
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/status')
def get_status():
    return jsonify(training_status)

if __name__ == '__main__':
    # Clear any existing training logs on startup
    for model in ['model1', 'model2']:
        try:
            Path(f'training_logs_{model}.json').unlink()
        except FileNotFoundError:
            pass
            
    app.run(debug=True, use_reloader=False)  # Disable reloader to prevent duplicate threads
 
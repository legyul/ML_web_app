import os
import io
import json
import cloudpickle
import pickle
import importlib
import subprocess
import sys

def install_and_import(package):
    '''
    If there are no pacages, install them and import
    '''
    try:
        return importlib.import_module(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return importlib.import_module(package)
    
def load_model(model_path):
    '''
    Load the model and automatically install the library

    Parameters
    - model_path (str): Saved model file path (ex. "models/model.pkl")

    Returns
    - object: loaded model
    '''
    try:
        # Set model.pkl_info.json file path
        info_path = model_path.replace(".pkl", "_info.json")

        # Load the required library list and install it
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                model_info = json.load(f)

            required_packages = model_info.get("required_packages", [])

            for package in required_packages:
                try:
                    install_and_import(package)
                except Exception as e:
                    print(f"Error installing package {package}: {e}")
                    continue
        
        # Load model
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        return model
    except FileNotFoundError:
        print(f"Model file '{model_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error decoding the model info JSON file for '{model_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred while loading the model: {e}")
        return None

def save_model_with_info(model, model_name, required_packages=None):
    if required_packages is None:
        required_packages = []
    
    model_info = {
        "required_packages": required_packages
    }

    model_info_buffer = io.BytesIO()
    model_info_json = json.dumps(model_info)
    model_info_buffer.write(model_info_json.encode('utf-8'))
    model_info_buffer.seek(0)
    
    model_buffer = io.BytesIO()
    cloudpickle.dump(model, model_buffer)
    model_buffer.seek(0)
    
    return model_info_buffer, model_buffer
            
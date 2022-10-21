from pathlib import Path
import zipfile
import wandb

def download_data(path):
  path='ecg_simula/setup_weights and biases/ecg_25000.zip:v0'
  run = wandb.init()
  data = run.use_artifact(path, type='raw_data')
  data=data.download()
  # Setup path to data folder
  data_path = Path("data/")
  train_path = data_path / "train_data"
  # If the image folder doesn't exist, download it and prepare it... 
  if train_path.is_dir():
      print(f"{train_path} directory exists.")
  else:
      print(f"Did not find {train_path} directory, creating one...")
      train_path.mkdir(parents=True, exist_ok=True)
      # Unzip pizza, steak, sushi data
  with zipfile.ZipFile("/content/artifacts/ecg_25000.zip:v0/ecg_25000.zip", "r") as zip_ref:
      print(f"Unzipping data to folder...") 
      zip_ref.extractall(train_path)
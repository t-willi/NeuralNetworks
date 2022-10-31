def unzip(save_path=None,zip_path=None):
  from pathlib import Path
  import zipfile
  # Setup path to data folder
  if isinstance(save_path, str):
    train_path = Path(save_path)
  if save_path == None:
    train_path = Path("data/train_data")
  #train_path = data_path / "train_data"
  # If the image folder doesn't exist, download it and prepare it... 
  if train_path.is_dir():
      print(f"{train_path} directory exists.")
  else:
      print(f"Did not find {train_path} directory, creating one...")
      train_path.mkdir(parents=True, exist_ok=True)
  # Unzip 
  zip_path=Path(zip_path)
  with zipfile.ZipFile(zip_path, "r") as zip_ref:
      print(f"Unzipping data to folder...") 
      zip_ref.extractall(train_path)
      print("unzip is finished")
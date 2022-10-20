#Directory containing usefull functions for machine learning and NN.
You can use this function to download files using the raw link
def request_import(path=None,name=None):
  import requests
  from pathlib import Path
  request = requests.get(path)
  name=name+".py"
  with open(name,"wb") as f:
    f.write(request.content)

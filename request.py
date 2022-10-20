# -*- coding: utf-8 -*-
"""Request.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1X7hEhPeuRh86sipQZrn08ZDUKbPVnX3C
"""

def request(path=None,name=None):
  import requests
  from pathlib import Path
  request = requests.get(path)
  name=name+".py"
  with open(name,"wb") as f:
    f.write(request.content)
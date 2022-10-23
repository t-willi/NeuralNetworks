import torch
import pandas as pd

def get_pred(dataset=None,model=None):
  """
  Function takes a Tensor Dataset as input,first a random file from the dataset is selected,
  then the Tensor pair is recombined and shaped into a df-->df_Input. 
  X is used afterwards as input into the model. The predictions are safed as --> df_output.
  Both dataframes are now unscaled by 5011, the max value of the whole dataset.
  Then a tuple pair of input and output is returned.
  """
  import random
  limit=len(dataset)
  rand_idx=random.randint(0,limit)
  X,y=dataset[rand_idx]
  #need to combine tensors to make dataframe for plotting input and output side by side
  full_tensor=torch.cat((X,y.squeeze()))
  full_tensor=full_tensor.numpy()
  df_input=pd.DataFrame(full_tensor).T
  df_input.columns = ["R1","R2","Rv1","Rv2","Rv3","Rv4","Rv5","Rv6"]
  model.to("cpu")
  model.eval()
  with torch.inference_mode():
    output=model(X)
  output=output.detach().numpy()
  #unscale data
  df_output = pd.DataFrame(output,columns=["F2","Fv1","Fv2","Fv3","Fv4","Fv5","Fv6"])*5011
  df_input=df_input*5011
  return df_input,df_output
import torch
import pandas as pd
def get_pred_12lead(dataset=None,Set=None,model=None,random=True,upscale=None):
  """
  Function takes a Tensor Dataset as input,first a random file from the dataset is selected,
  then the Tensor pair is recombined and shaped into a df-->df_Input. 
  X is used afterwards as input into the model. The predictions are safed as --> df_output.
  Both dataframes are now unscaled by 5011, the max value of the whole dataset.
  Then a tuple pair of input and output is returned.
  """
  if random:
    import random
    limit=len(dataset)
    rand_idx=random.randint(0,limit)
    X,y=dataset[rand_idx]
  if random is False:
      X,y=Set
  #need to combine tensors to make dataframe for plotting input and output side by side
  full_tensor=torch.cat((X,y.squeeze()))
  full_tensor=full_tensor.numpy()
  df_input=pd.DataFrame(full_tensor).T
  df_input.columns = ["real_I","real_II","real_v1","real_v2","real_v3","real_v4","real_v5","real_v6"]
  model.to("cpu")
  model.eval()
  with torch.inference_mode():
    output=model(X)
  output=output.detach().numpy()
  output=output.squeeze().T
  #unscale data
  df_output = pd.DataFrame(output,columns=["pred_II","pred_v1","pred_v2","pred_v3","pred_v4","pred_v5","pred_v6"])
  #calculate the 4 missing leads
  df_input.insert(2, "real_III", df_input["real_II"] - df_input["real_I"])
  df_input.insert(3,"real_aVR",0.5*(df_input["real_I"] + df_input["real_II"]))
  df_input.insert(4,"real_aVL",(df_input["real_I"] -0.5) * df_input["real_II"])
  df_input.insert(5,"real_aVF",(df_input["real_II"] -0.5) * df_input["real_I"])
  df_output.insert(0,"real_I",df_input["real_I"])
  df_output.insert(2,"real_III",df_output["pred_II"] - df_output["real_I"])
  df_output.insert(3,"real_aVR",0.5*(df_output["real_I"] + df_output["pred_II"]))
  df_output.insert(4,"real_aVL",(df_output["real_I"] -0.5) * df_output["pred_II"])
  df_output.insert(5,"real_aVF",(df_output["pred_II"] -0.5) * df_output["real_I"])
  if upscale:
    df_input=df_input*upscale
    df_output=df_output*upscale
    
  return df_input,df_output

# lead III value = (lead II value) - (lead I value)
# lead aVR value = -0.5*(lead I value + lead II value)
# lead aVL value = lead I value - 0.5 * lead II value
# lead aVF value = lead II value - 0.5 * lead I value

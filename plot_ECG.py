
import pandas as pd
from pathlib import Path
def plotECG(df1=None,df2=None,title=None,pad_df2=True,path=None,createECG=True,scale=None):
  """
  takes two dataframes with identical columns, concats them and plots them as ecg using ecg_plot
  it also takes the first column of df1 and ads it to df1 if pad_df2 is True
  """
  index=["real1","realR2","realv1","realv2","realv3","realv4","realv5","realv6","real_lead1",
         "pred2","predv1","predv2","predv3","predv4","predv5","predv6"]
  if createECG==True:
    ecg_path=path
    if Path(ecg_path).is_dir():
        print(f"{ecg_path} directory exists.")
    else:
        print(f"Did not find {ecg_path} directory, creating one...")
        Path(ecg_path).mkdir(parents=True, exist_ok=True)
    import ecg_plot
  if pad_df2 is True:
    if len(df1.columns)>len(df2.columns):
      df2.insert(0, 'real_lead1', df1["R1"])
  if scale:
    frames=[df1/1000,df2/1000]
  if scale is None:
    frames=[df1,df2]
  combined_df=pd.concat(frames,axis=1,join="outer",)
  if createECG is True:
    ecg_plot.plot(combined_df.values.T, sample_rate = 500,title = title,
                      lead_index = index )
    ecg_plot.save_as_png('ecg',ecg_path+"/")
  return combined_df

  

import pandas as pd

def plotECG(df1=None,df2=None,title=None,pad_df2=True,path=None):
  """
  takes two dataframes with identical columns, concats them and plots them as ecg using ecg_plot
  it also takes the first column of df1 and ads it to df1 if pad_df2 is True
  """
  index=["real1","realR2","realv1","realv2","realv3","realv4","realv5","realv6","real_lead1",
         "pred2","predv1","predv2","predv3","predv4","predv5","predv6"]
  import ecg_plot
  if pad_df2 is True:
    if len(df1.columns)>len(df2.columns):
      df2.insert(0, 'real_lead1', df1["R1"])
  frames=[df1/1000,df2/1000]
  combined_df=pd.concat(frames,axis=1,join="outer",)
  ecg_plot.plot(combined_df.values.T, sample_rate = 500,title = title,
                     lead_index = index )
  path.mkdir(parents=True, exist_ok=True)
  ecg_plot.save_as_png('ecg',path+"/")

  return combined_df


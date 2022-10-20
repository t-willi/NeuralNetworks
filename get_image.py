import matplotlib.pyplot as plt

def get_img(df,epoch,path):
  df=df
  columns=df.columns
  fig,axs = plt.subplots(7,1,figsize=(50,15))
  #plt.figure(figsize=(10,6))
  plt.suptitle(f"leads 2 to 8 from epoch{epoch}",x=0.5, y=0.93, fontsize=17, fontweight='700')
  for i,column in enumerate(columns):
    axs[i].plot(df[column], 'g', linewidth=2)
    axs[i].set_title(f'Plot {i}: lead {column}', fontsize=15)
  plt.xlabel('X[time in Datapoints]', fontsize=15, fontweight='bold')
  plt.ylabel('Y[mV]', fontsize=15, y=1, fontweight='bold')
  fig.savefig(path)   # save the figure to file
  plt.close(fig)

def find_max_min(data_dir=None):
  """
  This takes a directory of pd.read_csv readable data and intterates trough all,
  finding the global max and min
  """
  from tqdm.auto import tqdm
  max_list=[]
  min_list=[]
  #data_dir_train = "/content/data/train_data/sub_data/train"
  files=glob.glob(data_dir + '/*.asc')
  #files_train=glob.glob(data_dir_train + '/*.asc')
  #files=[*files_test,*files_train]
  for file in tqdm(files):
    temp_df=pd.read_csv(file,sep=" ",names=["1","2","v1","v2","v3","v4","v5","v6"])
    maximum=temp_df.max().max()
    minimum=temp_df.min().min()
    max_list.append(maximum)
    min_list.append(minimum)
    maximum=max(max_list)
    minimum=min(min_list)
  return(maximum,minimum)
  
def get_pred(dataset=val_dataset):
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
    output=torch.reshape(output,(5000, 7))
  output=output.detach().numpy()
  #unscale data
  df_output = pd.DataFrame(output,columns=["F2","Fv1","Fv2","Fv3","Fv4","Fv5","Fv6"])*5011
  df_input=df_input*5011
  return df_input,df_output
  
def plotECG(df1=None,df2=None,title=None,pad_df2=True,Path=None):
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
  df=pd.concat(frames,axis=1,join="outer",)
  ecg_plot.plot(df.values.T, sample_rate = 500,title = title,
                     lead_index = index )
  return df
    
    
def train_model(Epochs=20,model=model,train_dataloader=train_dataloader,val_dataloader=val_dataloader,stop=False):
  MODEL_NAME = "AE_test_wb"
  cfg = wandb.config
  model_path = Path("/content/model/output")
  model_path.mkdir(parents=True, exist_ok=True)
  # epoch_count = []
  # test_count = []
  # train_count = []
  from tqdm.auto import tqdm
  #train and test loop
  for epoch in tqdm(range(Epochs)):
    print(f"Epoch:{epoch}")
    train_loss=0
    for batch, (X,y) in enumerate(tqdm(train_dataloader)):
      #print("doing train loop")
      X, y = X.to(device), y.to(device) 
      model.train()
      output=model(X)
      output=torch.reshape(output,(128, 1, 7, 5000))
      #print(output.shape,y.shape)
      loss = criterion(output,y)
      #print(loss)
      train_loss += loss
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      if stop is not False:
        if batch == stop:
          break
      #print(train_loss)
    #average loss per batch
    train_loss /= len(train_dataloader)
    #start testing
    val_loss = 0
    model.eval()
    with torch.inference_mode():
      for batch,(X,y) in enumerate(val_dataloader):
        #print("doing test loop")
        X, y = X.to(device), y.to(device)
        val_pred = model(X)
        val_pred=torch.reshape(val_pred,(128, 1, 7, 5000))
        val_loss += criterion(val_pred,y)   
      if stop is not False:
        if batch == stop:
          break
      val_loss /= len(val_dataloader)
    #logging train and val los to w&b
    if epoch % 1 == 0:
      #print(f"\nTrain loss: {train_loss:.5f} |test_loss:{test_loss}" )
      # epoch_count.append(epoch)
      # test_count.append(test_loss.item())
      # train_count.append(train_loss.item())
      wandb.log({"train_loss": train_loss,
                "val_loss":val_loss})
      wandb.watch(model)
    #safing model to w%b
    if epoch % 1 == 0:
      model_path.mkdir(parents=True, exist_ok=True)
      torch.save(model.state_dict(),model_path/MODEL_NAME )
      trained_model_artifact = wandb.Artifact(
            MODEL_NAME, type="model",
            description="test run for w&b",
            metadata=dict(cfg))
      trained_model_artifact.add_dir(model_path)
      run.log_artifact(trained_model_artifact)
    #   training_progress = pd.DataFrame(
    # {'Epoch': epoch_count,
    #  'Train_loss': train_count,
    #  'Test_loss': test_count
    # })
    #   training_progress.to_csv("/content/gdrive/MyDrive/Simula/model_outcome/progress.csv")
    #saving prediction csv files and prediction img files to WB
    if epoch % 1 == 0:
      predictions = get_pred(test_dataset)
      predictions.to_csv(model_path/"prediction.csv")
      pred_table = wandb.Artifact("Prediction_table", type="Table")
      pred_table.add_file(model_path/"prediction.csv")
      run.log_artifact(pred_table)
      get_img(df=predictions,epoch=epoch,path=model_path/"img.png")
      img_artifact=wandb.Artifact("Prediction_Images",type="Image")
      img_artifact.add_file(model_path/"img.png")
      run.log_artifact(img_artifact)



def validation_loop(dataloader,batchsize=128):
    val_loss = 0
    model.eval()
    with torch.inference_mode():
      for batch,(X,y) in enumerate(dataloader):
        #print("doing test loop")
        X, y = X.to(device), y.to(device)
        val_pred = model(X)
        val_pred=torch.reshape(val_pred,(batchsize, 1, 7, 5000))
        val_loss += criterion(val_pred,y)   
      val_loss /= len(val_dataloader)
      return val_loss
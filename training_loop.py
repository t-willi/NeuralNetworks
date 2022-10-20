
def training_loop(dataloader=None,batchsize=128):
    for epoch in tqdm(range(Epochs)):
        print(f"Epoch:{epoch}")
        train_loss=0
        for batch, (X,y) in enumerate(tqdm(dataloader)):
            X, y = X.to(device), y.to(device) 
            model.train()
            output=model(X)
            output=torch.reshape(output,(batchsize, 1, 7, 5000))
            #print(output.shape,y.shape)
            loss = criterion(output,y)
            #print(loss)
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return train_loss
   
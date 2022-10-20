### This does not yet wotk cause i dont know how to call the 
# custom dataset class when i import the load _data function


def load_data(dir=None,BATCH_SIZE=128,Custom_dataset=None):
    train_dataset = Custom_dataset(dir=dir,split=True,target="train")
    test_dataset = Custom_dataset(data_dir=dir,split=True,target="test")
    val_dataset = Custom_dataset(dir=dir,split=True,target="val")
    #len(train_dataset),len(test_dataset),len(val_dataset)

    from torch.utils.data.dataloader import DataLoader
    #turn datasets into iterables
    train_dataloader = DataLoader(train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                drop_last=True
                                )
    val_dataloader = DataLoader(val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                drop_last=True
                                )

    return(train_dataloader,val_dataloader,test_dataset)
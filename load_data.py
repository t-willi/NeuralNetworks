def load_data(dir=None,BATCH_SIZE=128)
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
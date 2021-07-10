class data_config:
    data_dir = "./data"
    train_batch_size = 12
    test_batch_size = 12
    val_batch_size = 12
    num_workers = 6


class model_config:
    embed_size = 300
    hidden_size = 256
    attention_size = 256


class train_config:
    log_dir = "./runs"
    checkpoint_path = "./checkpoints"
    continue_training = False
    checkpoint_epochs = 1
    epochs = 100

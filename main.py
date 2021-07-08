from modelutils import load_checkpoint
from trainer import Trainer
from configs import data_config, model_config, train_config
import sys
import torch
from torch.optim.adam import Adam
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from collate import CollateText
from dataset import CNNDailyMail
from moduls.model import EncoderDecoder


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    cudnn.benchmark = True
    writer = SummaryWriter(log_dir=train_config.log_dir)

    # 1. Prepare the Data.
    vocab = torch.load('{}/vocab.pth'.format(data_config.data_dir))
    vocab_size = len(vocab.itos)

    train = CNNDailyMail(vocab=vocab, type1="train")
    test = CNNDailyMail(vocab=vocab, type1="test")
    val = CNNDailyMail(vocab=vocab, type1="validation")

    collate_fn = CollateText(batch_first=True, padding_value=0)

    train_dl = DataLoader(train, batch_size=data_config.train_batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=data_config.num_workers, drop_last=True)

    test_dl = DataLoader(test, batch_size=data_config.test_batch_size, shuffle=False,
                         collate_fn=collate_fn, num_workers=data_config.num_workers, drop_last=True)

    val_dl = DataLoader(val, batch_size=data_config.val_batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=data_config.num_workers, drop_last=True)

    # 2. Define the Model.
    model = EncoderDecoder(model_config=model_config,
                           vocab_size=vocab_size, device=device)

    # # 3. Train the Model.
    trainer = Trainer(train_dl, val_dl, writer, train_config)
    optimizer = Adam(model.parameters())
    last_epoch = -1
    min_val_loss = 1000
    model.to(device)

    if train_config.continue_training:
        path = "{}/model_checkpoint.pt".format(
            train_config.checkpoint_path)

        last_epoch, min_val_loss = load_checkpoint(
            path, model, optimizer)

        print("Loading last checkpoint with loss {} on epoch {}".format(
            min_val_loss, last_epoch))

    trainer.train(model, vocab, vocab_size, optimizer,
                  last_epoch, min_val_loss, device)

    # # 4. Evaluate the Model.
    # Evaluator().eval(model, test_dl, True, writer, "Test", device, vocab)

    writer.close()

    # 5. Make Predictions.
    # TODO


# def test():
#     a = torch.tensor([[1, 0, 0], [4, 5, 0], [7, 8, 9]])
#     ind = torch.tensor([1, 2, 3])

#     for i in range(len(ind)):
#         a[i][ind[i] - 1] = 0
#     print(a)


if __name__ == "__main__":
    # parse_args(sys.argv[1:])
    # test()
    main()

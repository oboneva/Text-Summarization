import torch
from configs import train_config
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import Module
from timeit import default_timer as timer
from modelutils import save_checkpoint


class Trainer:
    def __init__(self, train_dataloader: DataLoader, validate_dataloader: DataLoader, writer, configs: train_config):
        self.train_dl = train_dataloader
        self.val_dl = validate_dataloader
        self.epochs = configs.epochs
        self.writer = writer

        self.min_val_loss = 100
        self.no_improvement_epochs = 0
        self.patience = 10

        self.checkpoint_epochs = configs.checkpoint_epochs

    @torch.no_grad()
    def eval_loss(self, model: Module, dl: DataLoader, vocab_size: int, device):
        loss_func = nn.CrossEntropyLoss(ignore_index=0)
        loss = 0
        total_items = 0

        model.eval()
        for step, (text_padded, text_len, summary_padded, summary_len) in enumerate(dl):
            text_padded = text_padded.to(device)
            text_len = text_len.to(device)
            summary_padded = summary_padded.to(device)
            summary_len = summary_len.to(device)

            total_items += text_padded.size(0)

            output = model(text_padded, text_len, summary_padded, summary_len)
            # remove <sos>
            summary_padded = summary_padded[:, 1:]

            output = output.view(-1, vocab_size)
            summary_padded = summary_padded.reshape(-1)

            loss = loss_func(output, summary_padded)

            loss += loss.item()

            if step % 50 == 0:
                print("Loss/val at step {} {}".format(step, loss.item()))

        loss /= total_items

        return loss

    def train(self, model: Module, vocab, vocab_size: int, optimizer, start_epoch, min_val_loss, device):
        loss_func = nn.CrossEntropyLoss(ignore_index=0)  # index of <pad>
        self.min_val_loss = min_val_loss

        model.train()
        for epoch in range(start_epoch + 1, self.epochs):
            print("--------------- Epoch {} --------------- ".format(epoch))

            train_loss = 0
            total_items = 0

            for step, (text_padded, text_len, summary_padded, summary_len) in enumerate(self.train_dl):
                begin = timer()
                optimizer.zero_grad()

                text_padded = text_padded.to(device)
                text_len = text_len.to(device)
                summary_padded = summary_padded.to(device)
                summary_len = summary_len.to(device)

                total_items += text_padded.size(0)

                output = model(text_padded, text_len,
                               summary_padded, summary_len)

                # remove <sos>
                summary_padded = summary_padded[:, 1:]

                output = output.view(-1, vocab_size)
                summary_padded = summary_padded.reshape(-1)

                loss = loss_func(output, summary_padded)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                print("{0:.2f}".format(timer() - begin))

                if step % 10 == 0:
                    print("--------------- Step {} --------------- ".format(step))

                if step % 50 == 0:
                    print("Loss/train at step {} {}".format(step, loss.item()))

            train_loss /= total_items

            model.eval()

            # eval on the validation set
            val_loss = self.eval_loss(
                model, self.val_dl, vocab_size, device).item()

            # log loss
            print("MLoss/train", train_loss)
            # print("MLoss/validation", val_loss)

            self.writer.add_scalar("MLoss/train", train_loss, epoch)
            # self.writer.add_scalar("MLoss/validation", val_loss, epoch)
            self.writer.flush()

            dataiter = iter(self.val_dl)
            text, text_len, summary_padded, summary_len = next(dataiter)

            encoder_outputs, _, (hidden_state_n, cell_state_n) = model.encoder(
                text[:1].to(device), text_len[:1].to(device))
            output = model.decoder.summarize(
                encoder_outputs, (hidden_state_n, cell_state_n), vocab, 20)

            generated = ' '.join(output)
            print("Generated: ", generated)

            asd = [vocab.itos[num]
                   for num in summary_padded[:1].squeeze()[:summary_len[:1]]]
            references = ' '.join(asd)
            print("Reference: ", references)

            model.train()

            # early stopping
            if val_loss < self.min_val_loss:
                self.min_val_loss = val_loss
                self.no_improvement_epochs = 0

                print("New minimal validation loss", val_loss)

                path = "{}/model_best_state_dict.pt".format(
                    train_config.checkpoint_path)

                torch.save(model.state_dict(), path)

            elif self.no_improvement_epochs == self.patience:
                print("Early stopping on epoch {}".format(epoch))

                break
            else:
                self.no_improvement_epochs += 1

            # save checkpoint
            if epoch % self.checkpoint_epochs == 0:
                print("Saving checkpoint at {} epoch".format(epoch))

                path = "{}/model_checkpoint.pt".format(
                    train_config.checkpoint_path)

                save_checkpoint(model=model, optimizer=optimizer,
                                epoch=epoch, loss=self.min_val_loss, path=path)


def main():
    pass


if __name__ == "__main__":
    main()

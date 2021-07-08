
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from torch.utils.data import Dataset
from datasets.load import load_dataset


class CNNDailyMail(Dataset):
    def __init__(self, vocab: Vocab, type1: str):
        self.dataset = load_dataset('cnn_dailymail', '3.0.0',
                                    split='{}[:10%]'.format(type1))

        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = vocab

    def __len__(self):
        return len(self.dataset)

    def convert_strings_to_ints(self, strings):
        strings_vec = [self.vocab.stoi['<sos>']]
        for token in self.tokenizer(strings):
            strings_vec.append(
                self.vocab.stoi[token] if token in self.vocab.stoi else self.vocab.stoi['<unk>'])
        strings_vec.append(self.vocab.stoi['<eos>'])

        return strings_vec

    def __getitem__(self, index):
        article = self.dataset[index]["article"]
        highlights = self.dataset[index]["highlights"]

        article_vec = self.convert_strings_to_ints(article)
        article_vec = torch.tensor(article_vec)

        highlights_vec = self.convert_strings_to_ints(highlights)
        highlights_vec = torch.tensor(highlights_vec)

        return (article_vec, highlights_vec)


def main():
    pass


if __name__ == "__main__":
    main()

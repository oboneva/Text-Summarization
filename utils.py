from collections import Counter
from datasets import load_dataset
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab


def main():
    train_ds, test_ds, val_ds = load_dataset('cnn_dailymail', '3.0.0', split=[
                                             'train[:3%]', 'test[:3%]', 'validation[:3%]'])

    counter = Counter()
    tokenizer = get_tokenizer("basic_english")

    for item in val_ds:
        article = tokenizer(item["article"])
        highlights = tokenizer(item["highlights"])

        counter.update(article)
        counter.update(highlights)

    for item in test_ds:
        article = tokenizer(item["article"])
        highlights = tokenizer(item["highlights"])

        counter.update(article)
        counter.update(highlights)

    for item in train_ds:
        article = tokenizer(item["article"])
        highlights = tokenizer(item["highlights"])

        counter.update(article)
        counter.update(highlights)

    # TODO: revisit vocab, cause its size is actually ~ 200 000 tokens
    vocab = Vocab(counter, max_size=20000, min_freq=10, specials=[
        '<pad>', '<unk>', '<eos>', '<sos>'])

    torch.save(vocab, './vocab.pth')


if __name__ == "__main__":
    main()

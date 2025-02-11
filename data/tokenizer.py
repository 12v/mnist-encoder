import torch

vocab = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "<start>",
    "<finish>",
    "<pad>",
]


def tokenize_input_labels(labels):
    labels = labels.tolist()
    sentence = ["<start>"] + [str(label) for label in labels]
    return tokenize(sentence)


def tokenize_output_labels(labels):
    labels = labels.tolist()
    sentence = [str(label) for label in labels] + ["<finish>"]
    return tokenize(sentence)


def tokenize(labels):
    return torch.tensor([vocab.index(c) for c in labels], dtype=torch.long)

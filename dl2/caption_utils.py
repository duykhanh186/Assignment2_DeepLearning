import torch
from torch.utils.data import Dataset


SPECIAL_TOKENS = [
    "<NULL>",
    "<START>",
    "<END>",
]

BASE_WORDS = [
    "this",
    "image",
    "shows",
    "a",
    "the",
    "object",
    "is",
    "category",
]

CLASS_WORDS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def build_caption_vocab():
    words = SPECIAL_TOKENS + BASE_WORDS + CLASS_WORDS
    word_to_idx = {
        word: i
        for i, word in enumerate(words)
    }
    idx_to_word = {
        i: word
        for word, i in word_to_idx.items()
    }
    return word_to_idx, idx_to_word


def class_caption_tokens(class_name, variant=0):
    """
    Deterministic toy captions.

    These captions are intentionally simple: the goal is to practice
    the image-feature -> sequence-generation pipeline, not to build a
    state-of-the-art captioning dataset.
    """
    templates = [
        ["<START>", "this", "image", "shows", "a", class_name, "<END>"],
        ["<START>", "the", "object", "is", "a", class_name, "<END>"],
        ["<START>", "category", "is", class_name, "<END>"],
    ]
    return templates[variant % len(templates)]


def encode_caption(tokens, word_to_idx, max_length=8):
    null = word_to_idx["<NULL>"]

    ids = [
        word_to_idx[token]
        for token in tokens
    ]

    if len(ids) > max_length:
        ids = ids[:max_length]
        ids[-1] = word_to_idx["<END>"]

    ids = ids + [null] * (max_length - len(ids))

    return torch.tensor(
        ids,
        dtype=torch.long,
    )


class CIFARCaptionDataset(Dataset):
    """
    Wrap a CIFAR-10 dataset and attach a deterministic toy caption
    based on the image class.
    """

    def __init__(
        self,
        cifar_dataset,
        word_to_idx,
        max_length=8,
    ):
        self.dataset = cifar_dataset
        self.word_to_idx = word_to_idx
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        class_name = CLASS_WORDS[int(label)]

        tokens = class_caption_tokens(
            class_name,
            variant=idx,
        )

        caption = encode_caption(
            tokens,
            self.word_to_idx,
            max_length=self.max_length,
        )

        return image, caption, label


def decode_caption(ids, idx_to_word):
    words = []

    for idx in ids:
        token = idx_to_word[int(idx)]

        if token == "<START>":
            continue
        if token == "<END>":
            break
        if token == "<NULL>":
            continue

        words.append(token)

    return " ".join(words)

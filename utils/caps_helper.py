import numpy as np
from torchtext.vocab import GloVe


def get_case_id(word):
    if word.isupper():
        return 0  # all Caps
    elif word[0].isupper() and word[1:].islower():
        return 1  # upper Initial
    elif word.islower():
        return 2  # lower case
    elif any(c.isupper() for c in word) and not word[0].isupper():
        return 3  # mixed
    else:
        return 4


def load_glove_embeddings(word_to_idx, embedding_dim):
    glove = GloVe(name='6B', dim=embedding_dim)

    embeddings = np.random.uniform(-0.1, 0.1, (len(word_to_idx), embedding_dim))
    embeddings[0] = np.zeros(embedding_dim)  # Padding token at index 0

    for word, idx in word_to_idx.items():
        if word in glove.stoi:
            embeddings[idx] = glove[word].numpy()

    return embeddings
def build_vocab_from_dataset(dataset):
    words = set()
    for example in dataset:
        words.update(example['tokens'])
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for i, word in enumerate(sorted(words)):
        word_to_idx[word] = i + 2
    label_list = dataset.features['ner_tags'].feature.names
    label_to_idx = {label: i for i, label in enumerate(label_list)}
    return word_to_idx, label_to_idx

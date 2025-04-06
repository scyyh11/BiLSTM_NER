from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    sentences, labels = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    return sentences_padded, labels_padded


def collate_fn_caps(batch):
    sentences, labels, caps = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    case_padded = pad_sequence(caps, batch_first=True, padding_value=0)
    return sentences_padded, labels_padded, case_padded
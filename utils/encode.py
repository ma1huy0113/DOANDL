import torch

CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789 "
    "àáảãạăằắẳẵặâầấẩẫậ"
    "èéẻẽẹêềếểễệ"
    "ìíỉĩị"
    "òóỏõọôồốổỗộơờớởỡợ"
    "ùúủũụưừứửữự"
    "ỳýỷỹỵ"
    "đ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ"
    "ÈÉẺẼẸÊỀẾỂỄỆ"
    "ÌÍỈĨỊ"
    "ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ"
    "ÙÚỦŨỤƯỪỨỬỮỰ"
    "ỲÝỶỸỴ"
    "Đ"
)
char2idx = {c: i+1 for i, c in enumerate(CHARS)}  # 0 = blank (CTC)
idx2char = {i+1: c for i, c in enumerate(CHARS)}

unsupported = set()

def encode(text):
    encoded = []

    for c in text:
        if c in char2idx:
            encoded.append(char2idx[c])
        else:
            unsupported.add(c)

    return torch.tensor(encoded)

def decode(pred):
    result = []
    prev = -1
    for p in pred:
        if p != prev and p != 0:
            result.append(idx2char.get(int(p), ""))
        prev = p
    return "".join(result)



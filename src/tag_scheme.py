
from typing import List


def get_token_bio_from_strs(tokens: List[str]) -> List[str]:
    """
    for tokenization tasks
    """
    bios = []
    for tok in tokens:
        assert len(tok) != 0, "Contain invalid empty token: {} ".format(tokens)
        bios.append("B")
        _nchar_in_tok = len(tok)
        # token which has only one character
        if _nchar_in_tok == 1:
            continue
        bios.extend(["I" for _ in range(_nchar_in_tok - 2)])
        bios.append("O")
    _nchar = sum(len(tok) for tok in tokens)
    assert _nchar == len(bios), "Inconsistent length: {} != {}".format(_nchar, len(bios))
    return bios


def tokenize_from_bios(sentence: str, bios: List[str]) -> List[str]:
    assert len(sentence) == len(bios)
    tokens = []
    cur_token = ""
    for (char, tag) in zip(sentence, bios):
        if tag == "B":
            cur_token = char
        elif tag == "I":
            cur_token += char
        elif tag == "O":
            cur_token += char
            tokens.append(cur_token)
            cur_token = ""
        else:
            raise ValueError("Unexpected tag: {}".format(tag))

    assert sentence == "".join(tokens), "Fail to tokenize. BIO tags seems to hava invalid sequence" + \
                                        "input sentence={}, bio tags={}".format(sentence, bios)
    return tokens

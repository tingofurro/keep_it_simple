def cc_news_collate(inps):
    batch_paras = []
    for inp in inps:
        text = inp["text"]
        paragraphs = sorted(text.split("\n"), key=lambda p: abs(p.count(" ") - 35))
        batch_paras.append(paragraphs[0])
    return batch_paras


def cc_newsela_collate(inps):
    batch_paras = []
    for inp in inps:
        text = inp["p1"]
        paragraphs = sorted(text.split("\n"), key=lambda p: abs(p.count(" ") - 35))
        batch_paras.append(paragraphs[0])
    return batch_paras


def cnn_dailymail_collate(inps):
    batch_paras = []
    for inp in inps:
        text = inp["article"]
        paragraphs = sorted(text.split("\n"), key=lambda p: abs(p.count(" ") - 35))
        batch_paras.append(paragraphs[0])
    return batch_paras

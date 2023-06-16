def keyed_collate_fn(inps, key):
    batch_paras = []
    for inp in inps:
        text = inp.get(key)
        paragraphs = sorted(text.split("\n"), key=lambda p: abs(p.count(" ") - 35))
        batch_paras.append(paragraphs[0])
    return batch_paras

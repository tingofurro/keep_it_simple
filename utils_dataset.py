class CollateFn:
    def __init__(self, dataset_name: str):
        if dataset_name == "cc_news":
            self.key = "text"
        elif dataset_name == "newsela":
            self.key = "p1"
        elif dataset_name == "cnn_dailymail":
            self.key = "article"
        elif dataset_name == "xsum":
            self.key = "document"
        elif dataset_name == "imdb":
            self.key = "text"
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

    def collate_fn(self, inps):
        batch_paras = []
        for inp in inps:
            text = inp[self.key]
            paragraphs = sorted(text.split("\n"), key=lambda p: abs(p.count(" ") - 35))
            batch_paras.append(paragraphs[0])
        return batch_paras

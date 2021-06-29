from transformers import GPT2LMHeadModel, GPT2TokenizerFast, RobertaForSequenceClassification, RobertaTokenizerFast
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
import numpy as np, tqdm, json, collections, torch
from sklearn.metrics import f1_score
from torch.cuda.amp import autocast
from collections import Counter
import utils_optim

class FluencyRelativeScore:
    def __init__(self, same_length=False):
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.model.half().eval()
        self.same_length = same_length

    def preprocess_batch(self, decoded):
        # We cut short, but we want the end token at the end
        max_output_length = 80
        decs = [self.tokenizer.encode(dec) for dec in decoded]
        decs = [dec[:(max_output_length-1)] for dec in decs]

        decs_inp = torch.nn.utils.rnn.pad_sequence([torch.LongTensor([self.tokenizer.bos_token_id]+dec) for dec in decs], batch_first=True, padding_value=0)
        decs_out = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(dec+[self.tokenizer.eos_token_id]) for dec in decs], batch_first=True, padding_value=-1)
        return decs_inp.cuda(), decs_out.cuda()

    def text2loss(self, text, up_to_length=None):
        txt_inp, txt_out = self.preprocess_batch(text)

        if up_to_length is not None:
            txt_inp = txt_inp[:, :up_to_length]
            txt_out = txt_out[:, :up_to_length].contiguous()

        with torch.no_grad():
            model_outputs = self.model(input_ids=txt_inp, past_key_values=None)

            crit = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
            loss = crit(model_outputs["logits"].view(-1, self.tokenizer.vocab_size), txt_out.view(-1)).view(txt_out.shape)
            mask = (txt_inp != torch.LongTensor([0]).cuda()).float()
            non_pad_count = torch.sum(mask, dim=1)
            loss_per = torch.sum(loss, dim=1) / non_pad_count
        return loss_per

    def score(self, sources, generateds, partial=False, printing=False, **kwargs):
        up_to_length = None
        if self.same_length or partial:
            up_to_length = len(self.tokenizer.encode(generateds[0]))

        sources_score = self.text2loss(sources, up_to_length=up_to_length)
        generateds_score = self.text2loss(generateds, up_to_length=up_to_length)
        scores = (1.3 + sources_score - generateds_score) / 1.3
        scores = torch.clamp(scores, 0.001, 1.0).tolist()

        if printing:
            print("[fluency]", scores)
        return {"scores": scores, "sources_loss": sources_score, "generateds_loss": generateds_score}

class TextDiscriminator:
    def __init__(self, retrain_every=2000, fp16=False):
        # retrain_every: once the cache reaches that amount, the model is retrained.
        # fp16: Use half-precision for training

        self.fp16 = fp16

        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.discriminator = None
        self.optimizer = None
        self.optim_every = 2

        self.trained = False

        self.last_val_f1 = 0.0
        self.retrain_every = retrain_every
        self.cache_sources, self.cache_generateds = [], []

    def reload(self):
        # Reload everything
        torch.cuda.empty_cache()

        state_dict = torch.load(self.model_file)
        print(self.discriminator.load_state_dict(state_dict, strict=False))

        self.optimizer.state = collections.defaultdict(dict) # Reset state

    def train_from_dataset(self, texts, labels, n_epochs=1):
        toks = [torch.LongTensor(self.tokenizer.encode(text))[:200] for text in texts]
        toks = torch.nn.utils.rnn.pad_sequence(toks, batch_first=True, padding_value=0)

        train_batch_size = 8 if self.fp16 else 4

        dataset = TensorDataset(torch.LongTensor(toks), torch.LongTensor(labels))
        N_dev = min(100, int(0.1*len(dataset)))
        N_train = len(dataset) - N_dev

        d_train, d_dev = torch.utils.data.dataset.random_split(dataset, [N_train, N_dev])
        print("Num train: %d, num dev: %d; Label Count: %s" %(len(d_train), len(d_dev), str(Counter(labels))))

        train_sampler, dev_sampler = RandomSampler(d_train), RandomSampler(d_dev)

        train_dataloader = DataLoader(d_train, sampler=train_sampler, batch_size=train_batch_size)
        dev_dataloader = DataLoader(d_dev, sampler=dev_sampler, batch_size=50)

        # Start from scratch
        self.discriminator = None
        self.optimizer = None
        torch.cuda.empty_cache()

        self.discriminator = RobertaForSequenceClassification.from_pretrained("roberta-base").to("cuda")
        self.optimizer = utils_optim.build_optimizer(self.discriminator, learning_rate=1e-5)

        label_counter = Counter(labels)
        imbalance_weight = torch.FloatTensor([len(labels) / label_counter[0], len(labels) / label_counter[1]]).cuda()
        if self.fp16:
            imbalance_weight = imbalance_weight.half()
        print("Disc Imbalance Weights:", imbalance_weight.tolist())

        crit = torch.nn.CrossEntropyLoss(weight=imbalance_weight)

        best_state_dict = None
        best_f1 = 0.0
        for _ in range(n_epochs):
            print("New training epoch")
            self.discriminator.train()
            losses = []
            for i, batch in enumerate(tqdm.tqdm(train_dataloader)):
                batch_inputs, batch_labels = tuple(t.to("cuda") for t in batch)
                with autocast(self.fp16):
                    model_outputs = self.discriminator(batch_inputs) # , labels=batch_labels

                logits = model_outputs["logits"]
                loss = crit(logits, batch_labels)
                loss.backward()

                if i % self.optim_every == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                losses.append(loss.item())

            self.discriminator.eval()
            print("Train loss: %.3f" % (np.mean(losses)))
            with torch.no_grad():
                total_preds, total_labels = [], []
                for batch in dev_dataloader:
                    batch_inputs, batch_labels = tuple(t.to("cuda") for t in batch)
                    model_outputs = self.discriminator(batch_inputs)
                    preds = torch.argmax(model_outputs["logits"], axis=1).tolist()
                    total_labels += [l.item() for l in batch_labels]
                    total_preds += preds
                val_accuracy = np.mean(np.array(total_preds) == np.array(total_labels))
                val_f1 = f1_score(total_labels, total_preds, average="micro")

                if val_f1 >= best_f1:
                    best_state_dict = self.discriminator.state_dict()
                    best_f1 = val_f1
                print("Discriminator Validation. [Acc: %.3f] [F-1: %.3f]" % (val_accuracy, val_f1))

        self.discriminator.load_state_dict(best_state_dict)
        self.discriminator.eval()
        self.optimizer = None
        torch.cuda.empty_cache()

        total_preds, total_labels, total_pred_1s = [], [], []
        with torch.no_grad():
            for batch in dev_dataloader:
                batch_inputs, batch_labels = tuple(t.to("cuda") for t in batch)
                model_outputs = self.discriminator(batch_inputs)
                preds_probs = torch.nn.functional.softmax(model_outputs["logits"], dim=1)
                preds = torch.argmax(preds_probs, dim=1).tolist()
                total_labels += [l.item() for l in batch_labels]
                total_preds += preds
                prob_1s = preds_probs[:, 1]
                total_pred_1s += prob_1s.tolist()

            val_accuracy = np.mean(np.array(total_preds) == np.array(total_labels))
            val_f1 = f1_score(total_labels, total_preds, average="micro")

        print("[Final Discriminator] [Accuracy: %.3f] [F1: %.3f] [Average prediction: %.3f]" % (val_accuracy, val_f1, np.mean(total_pred_1s)))
        self.last_val_f1 = val_f1
        print("================")

    def retrain_auto(self):
        self.trained = True

        texts0 = list(set(self.cache_generateds))
        texts1 = list(set(self.cache_sources))
        print("[Disc] Number of negative samples: %d" % (len(texts0)))
        print("[Disc] Number of positive samples: %d" % (len(texts1)))

        texts = texts0 + texts1
        labels = ([0] * len(texts0)) + ([1] * len(texts1))

        self.train_from_dataset(texts, labels, n_epochs=3)

    def retrain_files(self, data_files, old_format=False):
        sentences, labels = [], []
        sentence_set = set([])
        for data_file in data_files:
            with open(data_file, "r") as f:
                for line in f:
                    obj = json.loads(line)
                    if obj['paragraph'] not in sentence_set:
                        sentence_set.add(obj['paragraph'])
                        sentences.append(obj['paragraph'])
                        labels.append(obj['label'])

        self.train_from_dataset(sentences, labels, n_epochs=5)
        return None

    def score(self, sources, generateds, partial=False, printing=False, **kwargs):
        if partial:
            # We don't do partial discrimination, wouldn't make sense...
            return {"scores": [1.0] * len(sources)}
        self.cache_sources += sources
        self.cache_generateds += generateds

        if len(set(self.cache_generateds) | set(self.cache_sources)) >= self.retrain_every:
            self.retrain_auto()
            self.cache_generateds = []
            self.cache_sources = []

        # If the model has not been trained yet
        if not self.trained:
            # Make it small but non-zero arbitrarily so that the multiplied score isn't nulled
            scores = torch.FloatTensor([0.2] * len(generateds)).cuda()
        else:
            # Do the actual scoring
            generateds = [text if len(text) > 0 else "empty text" for text in generateds] # Trying to fix the empty sequence problem
            toks = [torch.LongTensor(self.tokenizer.encode(text))[:200] for text in generateds]
            toks = [tok if len(tok) > 0 else [1] for tok in toks] # Make sure the sequence length is not zero, otherwise it crashes
            toks = torch.nn.utils.rnn.pad_sequence(toks, batch_first=True, padding_value=0).cuda()

            with torch.no_grad():
                model_outputs = self.discriminator(toks)
                probs = torch.nn.functional.softmax(model_outputs["logits"], dim=1)
                scores = torch.clamp(probs[:, 1], 0.0001, 1.0)

        scores = scores.tolist()
        if printing:
            print("[discriminator]", scores)

        return {"scores": scores, "val_f1": [self.last_val_f1] * len(scores)}

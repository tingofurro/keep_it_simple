from collections import Counter
import nltk, spacy, numpy as np
from rake_nltk import Rake

def get_pos_group(idx):
    if idx == 1:
        return ["AUX", "VERB", "PART"]
    elif idx == 2:
        return ["NOUN", "NUM", "PRON", "PROPN"]
    elif idx == 3:
        return ["ADJ", "ADV", "ADP", "INTJ", "SCONJ"]

def mask_tokens(tokens, kw_idxs, mask_idx=0):
    masked = [tok if tok not in kw_idxs else 0 for tok in tokens]
    is_masked = [0 if tok not in kw_idxs else 1 for tok in tokens]
    return masked, is_masked

def args2mask(args):
    if args.masking_strategy == "kw":
        print("Masking strategy kw: %.3f" % (args.kw_mask_ratio))
        masking_model = KeywordMasker(mask_ratio=args.kw_mask_ratio)
    elif args.masking_strategy == "pos":
        print("Masking strategy pos: %d" % (args.masking_pos_group))
        masking_model = POSMasker(get_pos_group(args.masking_pos_group))
    elif args.masking_strategy == "ratio":
        print("Masking strategy fixed ratio: %d; offset: %d" % (args.fixed_mask_ratio, args.fixed_mask_offset))
        masking_model = RatioMasker(k_ratio=args.fixed_mask_ratio, start_offset=args.fixed_mask_offset)
    elif args.masking_strategy == "nostop":
        print("Masking strategy non-stop words all masked")
        masking_model = NonStopMasker()
    return masking_model

def string2mask(masker_name):
    if masker_name[:2] == "kw":
        # kw30
        return KeywordMasker(mask_ratio=int(masker_name[2:])/100.0)
    elif masker_name[:3] == "pos":
        # pos1, pos2, pos3
        return POSMasker(get_pos_group(masker_name[3:]))
    elif masker_name[:5] == "ratio":
        # ratio2, ratio3.2
        rat, off = 2, 0
        if "." in masker_name:
            masker_name, off = masker_name.split(".")
        rat = int(masker_name[5:])
        return RatioMasker(k_ratio=rat, start_offset=int(off))
    elif masker_name == "nostop":
        return NonStopMasker()
    else:
        print("Could not match to a masker model")
        return None

class Masker:
    def __init__(self):
        self.model_tokenizer = None

    def register_tokenizer(self, tokenizer):
        self.model_tokenizer = tokenizer

    def compute_effective_mask_ratio(self, is_masked):
        return np.mean([np.mean(is_m) for is_m in is_masked])

class NonStopMasker(Masker):
    def __init__(self):
        # Masks everything but stop words
        self.stop_ws = set(nltk.corpus.stopwords.words("english"))

    def mask(self, sentences):
        unmasked, masked, is_masked = [], [], []

        for sentence in sentences:
            ums, ms, ims = [], [], []
            words = nltk.tokenize.word_tokenize(sentence)
            even = 0
            for w in words:
                toks = self.model_tokenizer.encode(" "+w, add_special_tokens=False)
                ums += toks
                even += 1
                if w.lower() not in self.stop_ws and even % 2 == 0:
                    ms += [0] * len(toks)
                    ims += [1] * len(toks)
                else:
                    ms += toks
                    ims += [0] * len(toks)
            unmasked.append(ums)
            masked.append(ms)
            is_masked.append(ims)
        return unmasked, masked, is_masked, self.compute_effective_mask_ratio(is_masked)

class KeywordMasker(Masker):
    def __init__(self, mask_ratio=0.2):
        self.stopws = set(nltk.corpus.stopwords.words("english") + [",", "''", "--", "-", ".", "(", ")", ";", "mr", "says", "say", "said", "will", "would"])
        self.r = Rake()
        self.mask_ratio = mask_ratio

    def compute_keywords(self, document):
        self.r.extract_keywords_from_text(document)
        kws = self.r.get_ranked_phrases_with_scores()

        word_scores = Counter()
        for c, kw in kws:
            for w in set(nltk.tokenize.word_tokenize(kw.lower())) - self.stopws:
                word_scores[w] += c

        final_keywords = [w for w, c in word_scores.most_common()]
        return final_keywords

    def mask_sentence(self, sentence, document_keywords):
        # new_sent = self.model_tokenizer.encode(sentence, add_special_tokens=False)

        words = nltk.tokenize.word_tokenize(sentence)
        num_to_mask = int((self.mask_ratio * len(words))+0.5)

        all_my_masks = sorted([w.lower() for w in words if w.lower() in document_keywords], key=lambda w: document_keywords.index(w))
        my_selected_masks = set(all_my_masks[:num_to_mask])

        ums, ms, ims = [], [], []
        for w in words:
            toks = self.model_tokenizer.encode(" "+w, add_special_tokens=False)
            ums += toks
            if w.lower() in my_selected_masks:
                ms += [0] * len(toks)
                ims += [1] * len(toks)
            else:
                ms += toks
                ims += [0] * len(toks)

        return ums, ms, ims

    def mask(self, sentences):
        assert self.model_tokenizer is not None, "Forgot to register the model tokenizer being used. Without it, it will not be possible to generate the outputs encoded for the model."
        unmasked, masked, is_masked = [], [], []

        if len(sentences) == 0:
            return [[0]], [[0]], [[0]], 0.0

        document = " ".join(sentences)
        document_kws = self.compute_keywords(document)

        for sentence in sentences:
            ums, ms, is_ms = self.mask_sentence(sentence, document_kws)
            unmasked.append(ums)
            masked.append(ms)
            is_masked.append(is_ms)

        return unmasked, masked, is_masked, self.compute_effective_mask_ratio(is_masked)

class POSMasker(Masker):
    def __init__(self, poses):
        # ADJ: adjective, ADP: adposition, ADV: adverb, AUX: auxiliary verb, CONJ: coordinating conjunction, DET: determiner, INTJ: interjection,
        # NOUN: noun, NUM: numeral, PART: particle, PRON: pronoun, PROPN: proper noun, PUNCT: punctuation, SCONJ: subordinating conjunction, SYM: symbol, VERB: verb
        self.poses = poses
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.remove_pipe("parser")
        self.nlp.remove_pipe("ner")

    def mask_sentence(self, sent_doc):
        # doc = self.nlp(sentence)
        unmasked, masked, is_masked = [], [], []
        for w in sent_doc:
            word_toks = self.model_tokenizer.encode(" "+w.text, add_special_tokens=False)
            unmasked += word_toks
            if w.pos_ in self.poses:
                masked += [0] * len(word_toks)
                is_masked += [1] * len(word_toks)
            else:
                masked += word_toks
                is_masked += [0] * len(word_toks)

        return unmasked, masked, is_masked

    def mask(self, sentences):
        assert self.model_tokenizer is not None, "Forgot to register the model tokenizer being used. Without it, it will not be possible to generate the outputs encoded for the model."
        unmasked, masked, is_masked = [], [], []

        if len(sentences) == 0:
            return [[0]], [[0]], [[0]], 0.0

        sent_docs = list(self.nlp.pipe(sentences, n_process=16))
        for sent_doc in sent_docs:
            ums, ms, is_ms = self.mask_sentence(sent_doc)
            unmasked.append(ums)
            masked.append(ms)
            is_masked.append(is_ms)

        return unmasked, masked, is_masked, self.compute_effective_mask_ratio(is_masked)

class RatioMasker(Masker):
    def __init__(self, k_ratio=3, start_offset=0):
        self.k_ratio = k_ratio
        self.start_offset = start_offset

    def mask_sentence(self, sentence, offset):
        words = self.model_tokenizer.encode(sentence, add_special_tokens=False)

        unmasked, masked, is_masked = [], [], []
        for i, w in enumerate(words):
            unmasked.append(w)
            if (i+offset) % self.k_ratio == 0:
                masked.append(0)
                is_masked.append(1)
            else:
                masked.append(w)
                is_masked.append(0)

        new_offset = (len(words)+offset) % self.k_ratio
        return unmasked, masked, is_masked, new_offset

    def mask(self, sentences):
        assert self.model_tokenizer is not None, "Forgot to register the model tokenizer being used. Without it, it will not be possible to generate the outputs encoded for the model."
        unmasked, masked, is_masked = [], [], []

        if len(sentences) == 0:
            return [[0]], [[0]], [[0]], 0.0

        offset = self.start_offset
        for sentence in sentences:
            ums, ms, is_ms, offset = self.mask_sentence(sentence, offset)
            unmasked.append(ums)
            masked.append(ms)
            is_masked.append(is_ms)

        return unmasked, masked, is_masked, self.compute_effective_mask_ratio(is_masked)

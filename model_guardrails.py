from collections import Counter
import nltk, spacy

class RelativeBrevityPenalizer:
    def __init__(self, min_ratio=0.6, max_ratio=1.5):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def compute_compression_ratio(self, source, generated):
        if source.count(" ") == 0:
            return 0.0
        return float(generated.count(" ")) / source.count(" ")

    def score(self, sources, generateds, partial=False, printing=False, **kwargs):
        brevity_penalties = []
        for source, decoded in zip(sources, generateds):
            brevity_penalty = 0.0
            compression_ratio = self.compute_compression_ratio(source, decoded)
            if partial:
                brevity_penalty = 1.0 if compression_ratio > self.max_ratio else 0.0
            else:
                brevity_penalty = 1.0 if compression_ratio < self.min_ratio or compression_ratio > self.max_ratio else 0.0
            brevity_penalties.append(brevity_penalty)
        if printing:
            print("[brevity]", brevity_penalties)
        return {"scores": brevity_penalties}

class RepeatNGramPenalty:
    def __init__(self, gram=3, keep_stop_ws=False):
        self.gram = gram
        self.stop_words = set(nltk.corpus.stopwords.words("english"))
        self.keep_stop_ws = keep_stop_ws

    def score(self, sources, generateds, **kwargs):
        repeat_penalties = []
        for generated in generateds:
            words = nltk.tokenize.word_tokenize(generated.lower())
            n_grams = [tuple(words[i:(i+self.gram)]) for i in range(len(words)-self.gram+1)]

            if not self.keep_stop_ws:
                n_grams = [ngram for ngram in n_grams if any(w not in self.stop_words for w in ngram)]

            n_repeated_three_grams = len([ngram for ngram, count in Counter(n_grams).most_common() if count > 1])
            repeat_penalty = 1.0 if n_repeated_three_grams > 0 else 0.0
            repeat_penalties.append(repeat_penalty)
        return {"scores": repeat_penalties}

def doc2ents(doc, black_list_types=[]):
    ents = [{"type": ent.label_, "entity": ent.text, "sent_idx": sent_idx} for sent_idx, sent in enumerate(doc.sents) for ent in sent.ents]
    ents = [e for e in ents if e["type"] not in black_list_types]
    return ents

class NERInaccuracyPenalty:
    def __init__(self, spacy_model="en_core_web_sm"):
        common_ents = ["one", "united states", "army"]

        self.common_ents = set([cent.lower() for cent in common_ents])
        self.word2num = {}
        self.black_list_types = set(["ORDINAL", "WORK_OF_ART", "EVENT", "PRODUCT", "LAW", "LANGUAGE"])
        self.number_words_to_remove = set(["the", "a", "an", "at", "of", "in", "than", "several", "few", "only", "about", "another", "least", "most", "last", "first", "early", "earlier",
                                           "later", "over", "fewer", "row", "every", "late", "ago", "only", "about", "around", "within", "more", "less"])

        self.string2digits = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
                              "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
                              "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "a hundred": 100, "hundred": 100, "a thousand": 1000,
                              "thousand": 1000}

        self.string2digits = {k: str(v) for k, v in self.string2digits.items()}
        self.digits2string = {v: k for k, v in self.string2digits.items()}
        self.spacy_model = spacy.load("en_core_web_sm")

    def common_ents_no_problem(self, ent_text):
        return ent_text in self.common_ents

    def clean_entity_text(self, ent_text):
        ent_text = ent_text.lower().replace("-", " ").replace('"', '').strip().replace("'s", "")
        if ent_text[:4] == "the ":
            ent_text = ent_text[4:]
        return ent_text.strip()

    def singular(self, ent_text):
        if len(ent_text) == 0:
            return ent_text
        if ent_text[-1] == "s":
            return ent_text[:-1]
        else:
            return ent_text

    def quantifier_cleaning(self, quantifier_text):
        words = nltk.tokenize.word_tokenize(quantifier_text.lower())
        words = sorted([w for w in words if len(w) >= 2 and w not in self.number_words_to_remove])
        return set(words)

    def quantifier_matching(self, quantifier, entity_list):
        quantifier_clean = self.quantifier_cleaning(quantifier)
        entity_list_clean = [self.quantifier_cleaning(ent["text"]) for ent in entity_list]
        return any([quantifier_clean in ent2_clean for ent2_clean in entity_list_clean])

    def remove_common_entities(self, ent_list_new, ent_list_old, source_text):
        source_text = source_text.lower()

        ent_set = set([self.clean_entity_text(e["text"]) for e in ent_list_old])
        finals = []

        for ent_new in ent_list_new:
            raw_entity_lower = ent_new["text"].lower()
            entity_text = self.clean_entity_text(ent_new["text"])
            if self.common_ents_no_problem(entity_text): # The entity is too common and could added anywhere
                continue
            if entity_text in ent_set or self.singular(entity_text) in ent_set: # Exact match with some entity
                continue
            if entity_text in source_text or self.singular(entity_text).lower() in source_text or raw_entity_lower in source_text:
                # Sometimes the NER model won't tag the exact same thing in the original paragraph, but we can just do string matching
                continue
            # Starting the entity-specific matching
            if ent_new["type"] in ["DATE", "CARDINAL", "MONEY", "PERCENT"]:
                # For dates:
                # a subset match is allowed: "several months" -> "months", "only a few weeks" -> "a few weeks"
                quantifier_clean = self.quantifier_cleaning(ent_new["text"])
                if self.quantifier_matching(ent_new["text"], ent_list_old):
                    continue

                if all([w in source_text for w in quantifier_clean]):
                    # A bit more desperate: remove additional words, and check that what's left is in the original
                    continue
                if ent_new["type"] == "CARDINAL":
                    if raw_entity_lower in self.string2digits and self.string2digits[raw_entity_lower] in source_text:
                        continue # They wrote "nineteen" instead of 19
                    elif raw_entity_lower in self.digits2string and self.digits2string[raw_entity_lower] in source_text.replace(",", ""):
                        continue # They wrote 19 instead of "nineteen"

            if ent_new["type"] == "GPE":
                if entity_text+"n" in ent_set or entity_text[:-1] in ent_set:
                    # If you say india instead of indian, or indian instead of india.
                    # Definitely doesn't work with every country, could use a lookup table
                    continue
            if ent_new["type"] in ["ORG", "PERSON"]:
                # Saying a smaller thing is fine: Barack Obama -> Obama. University of California, Berkeley -> University of California
                if any([entity_text in ent_text2 for ent_text2 in ent_set]):
                    continue
            finals.append(ent_new)
        return finals

    def score_one(self, ents1, ents2, source):
        new_ents2 = self.remove_common_entities(ents2, ents1, source)
        score = 1.0 if len(new_ents2) > 0 else 0.0
        return {"score": score, "new_ents": new_ents2, "gen_entities": ents2, "source_entities": ents1}

    def extract_entities(self, text):
        doc = self.spacy_model(text)
        return [{"text": ent.text, "type": ent.label_} for ent in doc.ents]

    def score(self, sources, generateds, printing=False, **kwargs):
        source_ents = [self.extract_entities(text) for text in sources]
        generated_ents = [self.extract_entities(text) for text in generateds]

        scores, all_new_ents = [], []
        for source_ent, generated_ent, source in zip(source_ents, generated_ents, sources):
            out = self.score_one(source_ent, generated_ent, source)
            scores.append(out["score"])
            all_new_ents.append(out["new_ents"])

        return {"scores": scores, "source_ents": source_ents, "gen_ents": generated_ents, "new_ents": all_new_ents}

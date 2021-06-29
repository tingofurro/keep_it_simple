from wordfreq import zipf_frequency
import textstat, numpy as np, nltk, torch

# we have a target shift. If you go beyond that, you should get penalized, but at a slower rate (right_slope).
def shift_to_score(shift, target_shift, right_slope=0.25):
    if shift <= target_shift:
        score = shift / (target_shift+0.001)
    else:
        score = 1.0 - right_slope * (shift - target_shift) / (target_shift+0.001)
    return np.clip(score, 0, 1.0)

# Vocab (V) and Readability (R) Shift models
class SimplicityLexicalScore:
    def __init__(self, target_shift=0.4, word_change_ratio=0.1):
        self.target_shift = target_shift
        self.word_change_ratio = word_change_ratio # Number of words that we expect to be swapped

        self.stopws = set(nltk.corpus.stopwords.words("english") + ["might", "would", "``"])

    def word_score_func(self, w):
        return zipf_frequency(w, 'en', wordlist="large")

    def is_good_word(self, w):
        if "'" in w:
            return False
        if len(w) > 30 or len(w) == 1:
            return False
        if w.lower() in self.stopws:
            return False
        if all(c.isdigit() for c in w):
            return False
        return True

    def vocab_shift_score(self, txt1, txt2, printing=False):
        words1 = nltk.tokenize.word_tokenize(txt1)
        words2 = nltk.tokenize.word_tokenize(txt2)
        words1 = set([w.lower() for w in words1 if self.is_good_word(w)])
        words2 = set([w.lower() for w in words2 if self.is_good_word(w)])

        removed_words = words1 - words2
        added_words = words2 - words1
        target_n_words = int(self.word_change_ratio * txt1.count(" "))

        vocab_shift = 0.0
        if target_n_words == 0:
            vocab_shift = 1.0 # You're not expected to have done any shifts yet
        elif len(removed_words) > 0 and len(added_words) > 0:
            # The idea of this is that we should consider only the K most complicated words removed
            # And by what top K most complicated they were replaced with.
            # The idea being that adding a bunch of simple words, or removing simple words doesn't matter beyond a certain point.

            added_words_zipfs = [{"w": w, "zipf": self.word_score_func(w)} for w in added_words]
            removed_words_zipfs = [{"w": w, "zipf": self.word_score_func(w)} for w in removed_words]
            added_words_zipfs = sorted(added_words_zipfs, key=lambda x: x['zipf'])
            removed_words_zipfs = sorted(removed_words_zipfs, key=lambda x: x['zipf'])[:target_n_words]

            removed_avg_zipfs = np.mean([x['zipf'] for x in removed_words_zipfs[:target_n_words]])
            added_avg_zipfs = np.mean([x['zipf'] for x in added_words_zipfs[:min(target_n_words, len(removed_words_zipfs))]])
            if printing:
                print("Desired # word swaps: %d" % (target_n_words))
                print("[Avg Zipf: %.3f] Added words:" % (added_avg_zipfs), added_words_zipfs)
                print("[Avg Zipf: %.3f] Removed words:" % (removed_avg_zipfs), removed_words_zipfs)

            vocab_shift = (added_avg_zipfs - removed_avg_zipfs) * len(removed_words_zipfs) / target_n_words

        return vocab_shift, len(added_words), len(removed_words)

    def score(self, sources, generateds, partial=False, printing=False, **kwargs):
        scores = []
        vshifts = []
        n_adds, n_dels = [], []
        for source, generated in zip(sources, generateds):
            if partial:
                source = " ".join(source.split(" ")[:generated.count(" ")])

            vshift, n_add, n_del = self.vocab_shift_score(source, generated, printing=printing)
            score = shift_to_score(vshift, self.target_shift)

            vshifts.append(vshift)
            scores.append(score)
            n_adds.append(n_add)
            n_dels.append(n_del)

        scores = torch.FloatTensor(scores)
        scores = (0.3 + torch.clamp(scores, 0.05, 1.0) * 0.7).tolist()

        if printing:
            print("[vshift]", scores)
        return {"scores": scores, "n_w_adds": n_adds, "n_w_dels": n_dels, "vshifts": vshifts}

class SimplicitySyntacticScore:
    def __init__(self):
        pass

    def rsource2target_shift(self, rsource):
        # Basically, the more complicated it is, the more we have to simplify.
        # In the Newsela data, there's strong correlation between the start readability level (rsource) and the amount of shift.
        # The higher you start, the more you have to drop. This piecewise linear function approximates it pretty well. (This is for a target level drop of 2 Newsela versions)
        if rsource <= 4.0:
            return 0
        elif rsource <= 12.0:
            return (rsource-3) * 0.5

        return 4.5 + (rsource-12) * 0.83

    def readability_shift_score(self, txt1, txt2):
        score1 = textstat.flesch_kincaid_grade(txt1)
        score2 = textstat.flesch_kincaid_grade(txt2)
        return score1, score2

    def score(self, sources, generateds, partial=False, printing=False, **kwargs):
        scores = []
        rshifts, rsources, rtargets = [], [], []
        for source, generated in zip(sources, generateds):
            if partial:
                source = " ".join(source.split(" ")[:generated.count(" ")])

            rsource, rtarget = self.readability_shift_score(source, generated)
            rshift = rsource - rtarget
            target_shift = self.rsource2target_shift(rsource)

            score = shift_to_score(rshift, target_shift)
            rshifts.append(rshift)
            rsources.append(rsource)
            rtargets.append(rtarget)
            scores.append(score)

        scores = torch.FloatTensor(scores)
        scores = (0.05 + torch.clamp(scores, 0.02, 1.0) * 0.95).tolist()

        if printing:
            print("[rshift]", scores)
        return {"scores": scores, "rshifts": rshifts, "rsources": rsources, "rtargets": rtargets}

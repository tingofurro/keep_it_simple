from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.nn.modules.loss import CrossEntropyLoss
import torch, os, numpy as np, nltk
import utils_masking

def unfold(sent_toks, make_tensor=True):
    unfolded = [w for sent in sent_toks for w in sent]
    if make_tensor:
        unfolded = torch.LongTensor(unfolded)
    return unfolded

class CoverageModel:
    def __init__(self, masking_model, model_card="roberta-base", device="cuda", model_file=None, is_soft=False, normalize=False, fp16=False):
        self.model_card = model_card

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_card)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_card)

        self.eos_token_id = self.tokenizer.eos_token_id
        if self.tokenizer.eos_token_id is None:
            self.eos_token_id = 0

        if type(masking_model) == str:
            self.masking_model = utils_masking.string2mask(masking_model)
        else:
            self.masking_model = masking_model
        self.masking_model.register_tokenizer(self.tokenizer)

        self.vocab_size = self.tokenizer.vocab_size
        self.device = device
        self.fp16 = fp16
        self.mask_id = 0

        self.normalize = normalize
        self.is_soft = is_soft
        if is_soft:
            print("Coverage will be soft.")
        if self.fp16:
            self.model.half()

        self.model.to(self.device)
        if model_file is not None:
            self.reload_model(model_file)

    def reload_model(self, model_file):
        print(self.model.load_state_dict(torch.load(model_file), strict=False))

    def save_model(self, model_file):
        torch.save(self.model.state_dict(), model_file)

    def process_text(self, document):
        sentences = [" "+sent for sent in nltk.tokenize.sent_tokenize(document) if len(sent) > 0]
        unmasked, masked, is_masked, mr_eff = self.masking_model.mask(sentences)
        return unfold(unmasked), unfold(masked), unfold(is_masked), mr_eff

    def build_io(self, targets, generateds):
        N = len(targets)

        input_ids, labels, is_masked, mr_effs = [], [], [], []
        gen_toks = []

        for target, generated in zip(targets, generateds):
            unmasked, masked, is_ms, mr_eff = self.process_text(target)
            input_ids.append(masked)
            labels.append(unmasked)
            is_masked.append(is_ms)
            gen_toks.append(torch.LongTensor(self.tokenizer.encode(generated, add_special_tokens=False)))
            mr_effs.append(mr_eff)

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        is_masked = torch.nn.utils.rnn.pad_sequence(is_masked, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1)
        input_ids = input_ids[:, :250]
        is_masked = is_masked[:, :250]
        labels = labels[:, :250]

        gen_toks = torch.nn.utils.rnn.pad_sequence(gen_toks, batch_first=True, padding_value=0)
        gen_toks = gen_toks[:, :250]
        gen_targets = torch.LongTensor([-1]).repeat(gen_toks.shape)

        seps = torch.LongTensor([self.eos_token_id]).repeat(N, 1)
        seps_targets = torch.LongTensor([-1]).repeat(seps.shape)

        input_ids = torch.cat((gen_toks, seps, input_ids), dim=1)
        labels = torch.cat((gen_targets, seps_targets, labels), dim=1)
        is_masked = torch.cat((torch.zeros_like(gen_toks), torch.zeros_like(seps), is_masked), dim=1)

        labels = labels.to(self.device)
        input_ids = input_ids.to(self.device)
        is_masked = is_masked.to(self.device)

        return input_ids, is_masked, labels, mr_effs

    def train_batch(self, contents, summaries):
        input_ids, is_masked, labels, mr_effs = self.build_io(contents, summaries)

        outputs = self.model(input_ids)
        logits = outputs["logits"]
        cross_ent = CrossEntropyLoss(ignore_index=-1)
        loss = cross_ent(logits.view(-1, self.vocab_size), labels.view(-1))

        num_masks = torch.sum(is_masked, dim=1).float() + 0.1
        with torch.no_grad():
            preds = torch.argmax(logits, dim=2)
            accs = torch.sum(preds.eq(labels).long() * is_masked, dim=1).float() / num_masks

        return loss, accs.mean().item()

    def score(self, bodies, decodeds, **kwargs):
        score_func = self.score_soft if self.is_soft else self.score_hard
        unnorm_scores = score_func(bodies, decodeds, **kwargs)

        if self.normalize:
            empty_scores = score_func(bodies, [""] * len(bodies), **kwargs)
            zero_scores = np.array(empty_scores["scores"])

            norm_scores = {k: v for k, v in unnorm_scores.items()}
            norm_scores["scores"] = ((np.array(unnorm_scores["scores"]) - zero_scores) / (1.0 - zero_scores))
            norm_scores["scores"] = norm_scores["scores"].tolist()
            return norm_scores
        else:
            return unnorm_scores

    def score_hard(self, bodies, decodeds, **kwargs):
        self.model.eval()
        with torch.no_grad():
            input_ids_w, is_masked_w, labels_w, mr_effs = self.build_io(bodies, decodeds)

            outputs_w = self.model(input_ids_w)
            preds_w = torch.argmax(outputs_w["logits"], dim=2)
            num_masks_w = torch.sum(is_masked_w, dim=1).float() + 0.1
            accs_w = torch.sum(preds_w.eq(labels_w).long() * is_masked_w, dim=1).float() / num_masks_w

            #     input_ids_wo, is_masked_wo, labels_wo = self.build_io(bodies, [""] * len(bodies))
            #     outputs_wo, = self.model(input_ids_wo)
            #     preds_wo = torch.argmax(outputs_wo, dim=2)
            #     num_masks_wo = torch.sum(is_masked_wo, dim=1).float() + 0.1
            #     accs_wo = torch.sum(preds_wo.eq(labels_wo).long() * is_masked_wo, dim=1).float() / num_masks_wo
        scores = accs_w # - accs_wo
        scores = scores.tolist()
        return {"scores": scores, "mr_eff": mr_effs}

    def score_soft(self, bodies, decodeds, printing=False, **kwargs):
        input_ids_w, is_masked_w, labels_w, mr_effs = self.build_io(bodies, decodeds)
        scores = self.score_soft_tokenized(input_ids_w, is_masked_w, labels_w)

        if printing:
            print("[coverage]", scores)

        return {"scores": scores, "mr_eff": mr_effs}

    def score_soft_tokenized(self, input_ids_w, is_masked_w, labels_w):
        self.model.eval()
        with torch.no_grad():
            outputs_w = self.model(input_ids_w)
            outputs_probs_w = torch.softmax(outputs_w["logits"], dim=2)
            max_probs, _ = outputs_probs_w.max(dim=2)

            relative_probs_w = (outputs_probs_w.permute(2, 0, 1) / max_probs).permute(1, 2, 0)

            batch_size, seq_len = is_masked_w.shape
            t_range = torch.arange(seq_len)

            scores = []
            for seq_rel_probs, seq_labels, seq_is_masked in zip(relative_probs_w, labels_w, is_masked_w):
                selected_probs = (seq_rel_probs[t_range, seq_labels])*seq_is_masked
                soft_score = torch.sum(selected_probs) / (torch.sum(seq_is_masked)+0.1)
                scores.append(soft_score.item())

        return scores


if __name__ == "__main__":
    import utils_misc
    from utils_dataset import SQLDataset

    MODELS_FOLDER = os.environ["MODELS_FOLDER"]
    utils_misc.select_freer_gpu()

    # model_file = os.path.join(MODELS_FOLDER, "bert_coverage_google_cnndm_length15_1.bin")
    # coverage_model = CoverageModel(masking_model, model_card="bert-base-uncased", device="cuda", model_file=model_file, is_soft=True, fp16=True)

    model_file = os.path.join(MODELS_FOLDER, "coverage_roberta_kw30p.bin")
    masking_model = utils_masking.KeywordMasker(mask_ratio=0.4)
    coverage_model = CoverageModel(masking_model, model_card="roberta-base", device="cuda", is_soft=True, fp16=True, model_file=model_file)

    dataset = [d for d in SQLDataset("/home/phillab/data/newsela/newsela_paired_0.2.db") if d["version2"] - d["version1"] == 2]
    coverages = []
    for batch in utils_misc.batcher(dataset, batch_size=100, progress=True):
        batch_scores = coverage_model.score([d["p1"] for d in batch], [d["p2"] for d in batch])
        coverages += batch_scores["scores"]
        print("Size of coverages: %d and mean: %.3f" % (len(coverages), np.mean(coverages)))

    text = """To the chagrin of New York antiques dealers, lawmakers in Albany have voted to outlaw the sale of virtually all items containing more than small amounts of elephant ivory,
            mammoth ivory or rhinoceros horn. The legislation, which is backed by Gov. Andrew M. Cuomo, will essentially eliminate New York’s central role in a well-established, nationwide
            trade with an estimated annual value of $500 million.Lawmakers say the prohibitions are needed to curtail the slaughter of endangered African elephants and rhinos, which they say
            is fueled by a global black market in poached ivory, some of which has turned up in New York.The illegal ivory trade has no place in New York State, and we will not stand for individuals
            who violate the law by supporting it,” Mr. Cuomo said in a statement on Tuesday, during the debate on the bill.The bill was approved by the Assembly on Thursday, 97 to 2, and passed
            the Senate, 43 to 17, on Friday morning."""

    # summaries = [
    #     """Lawmakers in Albany have voted to outlaw the sale of elephant ivory, mammoth ivory or rhinoceros horn.
    # The legislation backed by Gov. Andrew M. Cuomo, will essentially eliminate New York's central role in the nationwide
    # trade with an estimated annual value of $500 million""",
    #     "Lawmakers in Albany have voted to outlaw the sale of elephant ivory.",
    #     "Law to Impose Tough Limits on Sales of Ivory Art",
    #     "Lawmakers in Albany don't like ivory.",
    #     "Forbidden ivory in New York.",
    #     "Bla bla bla kokoko",
    #     ""]
    # contents = [text] * len(summaries)

    # scores_hard = coverage_model.score_hard(contents, summaries)
    # scores_soft = coverage_model.score_soft(contents, summaries)

    # print(text[:300])
    # print("--------")

    # for body, summary, score_hard, score_soft, mr_eff in zip(contents, summaries, scores_hard['scores'], scores_soft['scores'], scores_soft["mr_eff"]):
    #     print("[Hard: %.3f; Soft: %.3f] %s" % (score_hard, score_soft, summary))

from transformers import GPT2LMHeadModel, GPT2TokenizerFast, BartTokenizerFast
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import utils_sampling, utils_scoring
import torch, tqdm, time, os

def pad(data, padval=0):
    return torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=padval)

class Generator:
    def __init__(self, model_card, max_input_length=300, seq2seq=False, max_output_length=25, device='cuda'):
        # `model_card` can be a pretrained model name, or a model_folder
        self.model_card = model_card

        if not seq2seq:
            self.tokenizer = GPT2TokenizerFast.from_pretrained(model_card)
            self.model = GPT2LMHeadModel.from_pretrained(model_card)
            self.tokenizer.pad_token = "!"
        else:
            if "bart" in model_card:
                self.tokenizer = BartTokenizerFast.from_pretrained(model_card)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_card)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_card)

        if "prophetnet" in self.model_card:
            self.tokenizer.eos_token = "[UNK]"
            self.tokenizer.bos_token = "[SEP]"

        self.start_id = self.tokenizer.bos_token_id
        if model_card == "facebook/bart-large-cnn":
            # Weird the decoder start_token they trained with is not the tokenizer.bos_token_id
            self.start_id = 2
        if "pegasus" in model_card:
            self.start_id = 0

        self.seq2seq = seq2seq

        self.model.to(device)
        self.device = device

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        self.model.train()
        self.mode = "train"

    def reload(self, from_file, strict=True):
        loaded_dict = torch.load(from_file)
        loaded_dict = {k.replace("module.module.", ""): v for k, v in loaded_dict.items()}
        print(self.model.load_state_dict(loaded_dict, strict=strict))

    def save(self, to_file):
        torch.save(self.model.state_dict(), to_file)

    def preprocess_input(self, texts):
        tokenizer_outs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding="longest")
        encs = tokenizer_outs["input_ids"]
        attention_mask = tokenizer_outs["attention_mask"]

        encs = encs[:, :self.max_input_length].to(self.device)
        attention_mask = attention_mask[:, :self.max_input_length].to(self.device)
        # print("Model1:",encs)
        return encs, attention_mask

    def preprocess_batch(self, encoded, decoded):
        encs = self.preprocess_input(encoded)

        decs = [self.tokenizer.encode(dec, add_special_tokens=False) for dec in decoded]

        decs = [dec[:(self.max_output_length-1)] for dec in decs] # We cut short, but we want the end token at the end

        decs_inp = pad([torch.LongTensor([self.start_id]+dec) for dec in decs], padval=0).to(self.device)
        decs_out = pad([torch.LongTensor(dec+[self.tokenizer.eos_token_id]) for dec in decs], padval=-1).to(self.device)
        return encs, decs_inp, decs_out

    def encode(self, encoded_texts):
        input_ids, attention_mask = encoded_texts

        if not self.seq2seq:
            model_outs = self.model(input_ids=input_ids, past_key_values=None)
            return model_outs["past_key_values"]
        else:
            encoder = self.model.get_encoder()
            encoder_outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
            # A = self.model.forward(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, output_attentions=True, return_dict=True)
            # seq2seq_tracker = {"encoder": (A["encoder_last_hidden_state"], A["encoder_hidden_states"])}
            return {"encoder_outputs": encoder_outputs, "attention_mask": attention_mask, "past": None}

    def decode(self, decoded_texts, past=None, decoded_targets=None):
        if not self.seq2seq:
            model_outs = self.model(input_ids=decoded_texts, past_key_values=past)
            return model_outs["logits"]
        else:
            B = self.model(decoded_texts, attention_mask=past["attention_mask"], encoder_outputs=past["encoder_outputs"], return_dict=True)
            return B["logits"]

    def decode_fast(self, decoded_so_far, past):
        if not self.seq2seq:
            model_outputs = self.model(input_ids=decoded_so_far[:, -1].view(-1, 1), past_key_values=past)
            return model_outputs["logits"], model_outputs["past_key_values"]
        else:
            decoder_input = self.model.prepare_inputs_for_generation(decoded_so_far, past["past"], past["attention_mask"], use_cache=True, encoder_outputs=past["encoder_outputs"])
            decoder_out = self.model(**decoder_input, return_dict=True)
            past["past"] = decoder_out.past_key_values

            return decoder_out.logits, past

    def toks2text_batch(self, tokens_batch, return_tokens=False):
        end_id = self.tokenizer.eos_token_id

        tokens_batch = [tokens[1:].tolist() + [end_id] for tokens in tokens_batch] # Add the end_id just in case
        tokens_batch = [tokens[:tokens.index(end_id)] for tokens in tokens_batch] # Cut at the end token

        # texts = [self.tokenizer.decode(tokens) for tokens in tokens_batch]
        texts = self.tokenizer.batch_decode(tokens_batch, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        if not return_tokens:
            return texts
        else:
            return texts, tokens_batch

    def train_batch(self, encoded, decoded=None, decoded_tokenized=None, no_preinput=False, input_past=None, return_logits=False):
        # if self.mode != 'train':
        #     print("BEWARE. Model is not in train mode.")
        assert decoded is not None or decoded_tokenized is not None, "Train batch should either receive decoded (list of text), or decoded_tokenized (list of list of token integers)."

        if decoded_tokenized is not None:
            # We are forcing to re-use pre-tokenized stuff
            encs = self.preprocess_input(encoded)
            decs_inp = pad([torch.LongTensor([self.start_id]+dec) for dec in decoded_tokenized], padval=0).to(self.device)
            decs_out = pad([torch.LongTensor(dec+[self.tokenizer.eos_token_id]) for dec in decoded_tokenized], padval=-1).to(self.device)
        else:
            encs, decs_inp, decs_out = self.preprocess_batch(encoded, decoded)

        if "prophetnet" in self.model_card:
            # Have to do something a bit different because it has its own n-gram los
            encs, attn_mask = encs
            model_outs = self.model(input_ids=encs, labels=decs_inp, return_dict=True)
            return model_outs["loss"]
        else:
            past = None
            if input_past is not None:
                past = input_past
            elif not no_preinput:
                past = self.encode(encs)

            crit = torch.nn.CrossEntropyLoss(ignore_index=-1)
            logits = self.decode(decs_inp, past=past)
            if return_logits:
                return logits
            loss = crit(logits.view(-1, self.tokenizer.vocab_size), decs_out.contiguous().view(-1))
            return loss

    def train(self):
        self.model.train()
        self.mode = 'train'

    def eval(self):
        self.mode = 'eval'
        self.model.eval()

    def generate_batch(self, encoded_texts, max_output_length=100, sample=False, force_start=None, temperature=1.0, top_k=0, top_p=1.0, no_copy_ngram=0, no_repeat_ngram=0, min_length=0, **kwargs):
        N = len(encoded_texts)

        force_start_ids = []
        if force_start is not None:
            force_start_ids = self.tokenizer.encode(force_start, add_special_tokens=False)

        if self.model_card == "facebook/bart-large-cnn":
            force_start_ids = [0]

        inputs = self.preprocess_input(encoded_texts)
        past = self.encode(inputs)

        build_up = torch.LongTensor([self.start_id]).repeat(N, 1).to(self.device)
        # print("WE ARE HERE: ", build_up)
        seq_logprobs = torch.zeros((N)).to(self.device)

        end_id = self.tokenizer.eos_token_id
        finished_func = lambda build_up: all([end_id in build for build in build_up[:, 1:]])

        while build_up.shape[1] <= max_output_length and not finished_func(build_up):
            is_force_start = len(force_start_ids) > 0 and build_up.shape[1] <= len(force_start_ids)

            logits, past = self.decode_fast(build_up, past)

            logits = logits.view(N, -1)

            logits = utils_sampling.ngram_copy_filtering(build_up, inputs[0], logits, n_gram=no_copy_ngram)
            logits = utils_sampling.ngram_copy_filtering(build_up, build_up, logits, n_gram=no_repeat_ngram)
            logits = utils_sampling.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

            if min_length > 0 and build_up.shape[1] <= min_length and not is_force_start:
                logits[:, end_id] -= 1000.0

            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

            if is_force_start:
                force_idx = build_up.shape[1]-1
                current = torch.LongTensor([force_start_ids[force_idx]]).repeat(N, 1).to(self.device)
            elif sample:
                probs = torch.nn.functional.softmax(logits/temperature, dim=-1).squeeze(1)
                distrib = torch.distributions.categorical.Categorical(probs)
                current = distrib.sample().unsqueeze(-1)
            else:
                current = torch.argmax(logprobs, dim=-1)

            current = current.view(-1, 1)
            build_up = torch.cat((build_up, current), dim=1)

            not_finished = (1-torch.any(build_up[:, 1:]==end_id, dim=1).float()).to(self.device)
            if not (self.model_card == "facebook/bart-large-cnn" and is_force_start): # otherwise we force pick an end token at the start
                seq_logprobs += not_finished * logprobs[torch.arange(N), current.view(N)].view(N)

        outputs = {}
        outputs['output_text'], outputs["output_tokens"] = self.toks2text_batch(build_up, return_tokens=True)
        outputs['logprob'] = seq_logprobs.tolist()

        outputs_list = [{k: outputs[k][i] for k in outputs} for i in range(N)]
        return outputs_list

    def generate_ckpt_batch(self, encoded_texts, scorer=None, max_output_length=100, ckpt_every=5, ckpt_runs=3, sample=False, force_start=None, temperature=1.0, top_k=0, top_p=1.0, no_copy_ngram=0, no_repeat_ngram=0,
                            printing=False, min_length=0, **kwargs):
        assert top_p == 1.0, "For now, the top_p implementation does not work, as the sampling on GPU will crash randomly"
        N = len(encoded_texts) * ckpt_runs

        # Used in the checkpointing
        encoded_toks = [self.tokenizer.encode(encoded_text, add_special_tokens=False) for encoded_text in encoded_texts]
        encoding_equivalents = [self.tokenizer.decode(encoded_tok) for encoded_tok in encoded_toks for _ in range(ckpt_runs)] # The unlooped version

        force_start_ids = []
        if force_start is not None:
            force_start_ids = self.tokenizer.encode(force_start, add_special_tokens=False)

        if self.model_card == "facebook/bart-large-cnn":
            force_start_ids = [0]

        inputs = self.preprocess_input(encoded_texts)
        past = self.encode(inputs)
        past = self.past_repeat_interleave(past, ckpt_runs)

        build_up = torch.LongTensor([self.start_id]).repeat(N, 1).to(self.device)
        inputs_repeated = torch.repeat_interleave(inputs[0], repeats=ckpt_runs, dim=0)

        scores = torch.zeros((N)).to(self.device)

        end_id = self.tokenizer.eos_token_id
        finished_func = lambda build_up: all([end_id in build for build in build_up[:, 1:]])

        while build_up.shape[1] < max_output_length and not finished_func(build_up):
            is_force_start = len(force_start_ids) > 0 and build_up.shape[1] <= len(force_start_ids)
            logits, past = self.decode_fast(build_up, past)
            logits = logits.view(N, -1)

            logits = utils_sampling.ngram_copy_filtering(build_up, inputs_repeated, logits, n_gram=no_copy_ngram)
            logits = utils_sampling.ngram_copy_filtering(build_up, build_up, logits, n_gram=no_repeat_ngram)
            if sample:
                logits = utils_sampling.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

            if min_length > 0 and build_up.shape[1] <= min_length and not is_force_start:
                logits[:, end_id] -= float("Inf")

            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

            if len(force_start_ids) > 0 and build_up.shape[1] <= len(force_start_ids):
                force_idx = build_up.shape[1]-1
                current = torch.LongTensor([force_start_ids[force_idx]]).repeat(N, 1).to(self.device)
            elif sample:
                probs = torch.nn.functional.softmax(logits/temperature, dim=-1).squeeze(1)
                distrib = torch.distributions.categorical.Categorical(probs)
                current = distrib.sample().unsqueeze(-1)
                # current = torch.multinomial(probs, 1)
            else:
                current = torch.argmax(logprobs, dim=-1)

            current = current.view(-1, 1)
            build_up = torch.cat((build_up, current), dim=1)
            not_finished = (1-torch.any(build_up[:, 1:]==end_id, dim=1).float()).to(self.device)
            scores += not_finished * logprobs[torch.arange(N), current.view(N)].view(N)

            if (build_up.shape[1]-1) % ckpt_every == 0:
                # NEED TO CHECKPOINT

                generated_so_far = self.toks2text_batch(build_up)
                if printing:
                    print("============== CKPT %d =================" % (build_up.shape[1]-1))
                    print("Options:")
                    for option in generated_so_far:
                        print(option)
                    print("-----------")

                so_far_scores = scorer(encoding_equivalents, generated_so_far, partial=True, printing=printing)
                so_far_scores = torch.FloatTensor(so_far_scores["total_scores"]).cuda()

                folded_scores = so_far_scores.view(-1, ckpt_runs)
                best_idx_in_each = torch.argmax(folded_scores, dim=-1)
                chosen_tracks = torch.arange(0, N, step=ckpt_runs).cuda() + best_idx_in_each
                chosen_tracks = torch.repeat_interleave(chosen_tracks, repeats=ckpt_runs, dim=0)

                # Pump it back up to k candidates; not seq2seq compatible for now
                build_up = build_up[chosen_tracks]
                scores = scores[chosen_tracks]
                past = self.past_beam_bookkeeping(past, chosen_tracks)

        # Clear outputs: keep only one for each run
        # First rescore, and then filter out one for each!
        encodings_repeated = [encoded_text for encoded_text in encoded_texts for _ in range(ckpt_runs)]

        final_scores = scorer(encodings_repeated, self.toks2text_batch(build_up))
        final_scores = torch.FloatTensor(final_scores["total_scores"]).cuda()
        folded_scores = final_scores.view(-1, ckpt_runs)
        chosen_tracks = torch.arange(0, N, step=ckpt_runs).cuda() + torch.argmax(folded_scores, dim=-1)
        # print(folded_scores, chosen_tracks)

        scores = scores[chosen_tracks]
        build_up = build_up[chosen_tracks]
        real_scores = final_scores[chosen_tracks]

        outputs = {}
        outputs["output_text"], outputs['output_tokens'] = self.toks2text_batch(build_up, return_tokens=True)

        outputs['logprob'] = scores.tolist()
        outputs["score"] = real_scores.tolist()

        outputs_list = [{k: outputs[k][i] for k in outputs} for i in range(len(encoded_texts))]
        return outputs_list

    def past_beam_bookkeeping(self, past, tracks):
        if not self.seq2seq:
            return [p[:, tracks, :] for p in past]
        else:
            # Get ready for some high-precision stitching...
            if past["past"] is not None:
                past["past"] = self.model._reorder_cache(past["past"], tracks)
            return past

    def past_repeat_interleave(self, past, beam_size):
        if not self.seq2seq:
            # seq_expand = lambda x:
            # print(">>>", past[0][0].shape, )
            return [torch.repeat_interleave(p, repeats=beam_size, dim=1) for p in past]
        else:
            past["attention_mask"] = torch.repeat_interleave(past["attention_mask"], repeats=beam_size, dim=0)
            past["encoder_outputs"] = tuple([torch.repeat_interleave(X, repeats=beam_size, dim=0) for X in past["encoder_outputs"]])
            return past

    def generate_beam_batch(self, bodies, beam_size=3, max_output_length=100, sample=False, temperature=1.0, top_k=0, top_p=1.0, no_copy_ngram=0, no_repeat_ngram=0, force_start=None, scorer=None, ckpt_every=0, printing=False, min_length=0, **kwargs):
        assert top_p == 1.0, "For now, the top_p implementation does not work, as the sampling on GPU will crash randomly"

        timings = False
        T = time.time()

        if timings:
            print("------------")

        force_start_ids = []
        if force_start is not None:
            force_start_ids = self.tokenizer.encode(force_start, add_special_tokens=False)

        if self.model_card == "facebook/bart-large-cnn":
            force_start_ids = [0]

        batch_size = len(bodies)
        N = batch_size * beam_size

        expanded_inputs = [enc_inp for enc_inp in bodies for _ in range(beam_size)]
        inputs = self.preprocess_input(bodies)

        if timings:
            print("tokenization", time.time()-T)
            T = time.time()

        build_up = torch.LongTensor([self.start_id]).repeat(N, 1).to(self.device)

        seq_logprobs = torch.zeros((N)).to(self.device)
        scores = torch.zeros((N)).to(self.device)

        one_every_k = torch.FloatTensor([1] + [0] * (beam_size-1)).repeat(batch_size*beam_size).to(self.device)

        # Sometimes, we process the same input, as we run it once as a sampled, and once as an argmax, in which case we should reuse the computation
        past = self.encode(inputs)

        # print("OVER HERE:", len(past), len(past[0]), past[0][0].shape)
        past = self.past_repeat_interleave(past, beam_size)
        inputs_repeated = torch.repeat_interleave(inputs[0], repeats=beam_size, dim=0)

        end_id = self.tokenizer.eos_token_id
        finished_func = lambda build_up: all([end_id in build for build in build_up[:, 1:]])
        next_force_split = False

        while build_up.shape[1] < max_output_length and not finished_func(build_up):
            is_force_start = len(force_start_ids) > 0 and build_up.shape[1] <= len(force_start_ids)
            logits, past = self.decode_fast(build_up, past)
            logits = logits.view(N, -1)

            logits = utils_sampling.ngram_copy_filtering(build_up, inputs_repeated, logits, n_gram=no_copy_ngram)
            logits = utils_sampling.ngram_copy_filtering(build_up, build_up, logits, n_gram=no_repeat_ngram)
            if sample:
                logits = utils_sampling.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

            if min_length > 0 and build_up.shape[1] <= min_length and not is_force_start:
                logits[:, end_id] -= float("Inf")

            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

            if is_force_start:
                force_idx = build_up.shape[1]-1
                all_selects = torch.LongTensor([force_start_ids[force_idx]]).repeat(N, beam_size).to(self.device)
            elif sample:
                probs = torch.nn.functional.softmax(logits/temperature, dim=-1)
                distrib = torch.distributions.categorical.Categorical(probs)

                all_selects = torch.cat([distrib.sample().unsqueeze(-1) for _ in range(beam_size)], dim=1)
                # torch.multinomial basically doesn't work in torch 1.6+ and in 1.5.1- it throws a bug when coupled with
                # all_selects = torch.multinomial(probs, beam_size)
            else:
                _, all_selects = torch.topk(logprobs, k=beam_size, dim=-1)

            not_finished = (1-torch.any(build_up[:, 1:]==end_id, dim=1).float()).to(self.device)
            expanded_not_finished = torch.repeat_interleave(not_finished, repeats=beam_size)

            expanded_logprobs = torch.repeat_interleave(seq_logprobs, repeats=beam_size) # This should be batch_size * beam_size²
            expanded_logprobs += expanded_not_finished * logprobs[torch.repeat_interleave(torch.arange(N), repeats=beam_size), all_selects.view(-1)]
            # expanded_logprobs += logprobs[torch.repeat_interleave(torch.arange(N), repeats=beam_size), all_selects.view(-1)]

            # We don't want you to select from finished beams
            expanded_logprobs -= (1-expanded_not_finished)*(1-one_every_k)*1000.0
            expanded_score = expanded_logprobs # This is if we don't have a scorer

            batched_logprobs = expanded_logprobs.view(batch_size, -1)
            batched_scores = expanded_score.view(batch_size, -1)

            if build_up.shape[1] == 1 or (len(force_start_ids) == build_up.shape[1]-1) or next_force_split:
                # print("Force splitting is going to happen")
                # Force the model to differ in path: on (1) the first token generated, or (2) the first token generated after the force_start track
                choices = torch.arange(beam_size, device=self.device).repeat(batch_size)
                batched_choices = choices.view(batch_size, beam_size)
                next_force_split = False
            else:
                _, batched_choices = torch.topk(batched_scores, k=beam_size, dim=1) # Going from k² choices per element to k choices.

            batched_tracks = batched_choices // beam_size
            tracks = beam_size*torch.repeat_interleave(torch.arange(batch_size), repeats=beam_size).to(self.device) + batched_tracks.view(-1)

            selected_scores = batched_scores[torch.repeat_interleave(torch.arange(batch_size), repeats=beam_size), batched_choices.view(-1)]
            selected_logprobs = batched_logprobs[torch.repeat_interleave(torch.arange(batch_size), repeats=beam_size), batched_choices.view(-1)]

            # Figure out the kept words to be added to the build-up
            per_batch_selects = all_selects.view(batch_size, -1)
            next_words = per_batch_selects[torch.repeat_interleave(torch.arange(batch_size), repeats=beam_size), batched_choices.view(-1)]
            next_words = next_words.unsqueeze(1)
            # print("ABC", next_words)

            not_finished = not_finished[tracks] # Rewire the not_finished
            next_words = (next_words * not_finished.view(-1, 1).long()) # + (1-not_finished.view(-1, 1).long()) * end_id # This is so that nothing gets written past end_token but other end_tokens

            # if build_up.shape[1] >= 82:
            #     print(build_up.shape[1], not_finished.long(), next_words)
            # print("AB",next_words.shape)
            # print(">>>>", next_words)

            # [BOOKKEEPING] Going from k² to k options at each time means we have to swap all the caches around: past, build-up
            # print("!!!", not_finished)
            # not_finished = not_finished[tracks] # Need to recompute of not_finished after reshuffle
            # print("<<<", not_finished)

            build_up = build_up[tracks, :]
            past = self.past_beam_bookkeeping(past, tracks)

            # Update the latest scores, and the current_build
            build_up = torch.cat((build_up, next_words), dim=1)
            # print("[NL]", build_up)

            # print("%d" % (build_up.shape[1]), self.tokenizer.batch_decode(build_up.tolist()))

            scores = selected_scores.view(-1)
            seq_logprobs = selected_logprobs.view(-1)

            if ckpt_every > 0 and (build_up.shape[1]-1) % ckpt_every == 0:
                # NEED TO CHECKPOINT
                generated_so_far = self.toks2text_batch(build_up)
                if printing:
                    print("============== CKPT %d =================" % (build_up.shape[1]-1))
                    print("Options:")
                    for option in generated_so_far:
                        print(option)
                    print("-----------")

                so_far_scores = scorer(expanded_inputs, generated_so_far, partial=True, printing=printing)
                so_far_scores = torch.FloatTensor(so_far_scores["total_scores"]).cuda()

                folded_scores = so_far_scores.view(-1, beam_size)
                best_idx_in_each = torch.argmax(folded_scores, dim=-1)
                chosen_tracks = torch.arange(0, N, step=beam_size).cuda() + best_idx_in_each
                chosen_tracks = torch.repeat_interleave(chosen_tracks, repeats=beam_size, dim=0)

                # Pump it back up to k candidates; not seq2seq compatible for now
                build_up = build_up[chosen_tracks]
                scores = scores[chosen_tracks]
                past = self.past_beam_bookkeeping(past, chosen_tracks)

                next_force_split = True

        if timings:
            print("loop", time.time()-T)
            T = time.time()

        # print("[NL]", build_up.shape, build_up)

        batched_build_up = build_up.view(batch_size, beam_size, -1)
        batched_logprobs = seq_logprobs.view(batch_size, -1)
        batched_scores = scores.view(batch_size, -1)

        outputs = []

        for orig_beams, beam_logprobs, beam_scores in zip(batched_build_up, batched_logprobs, batched_scores):
            output_txts, beams = self.toks2text_batch(orig_beams, return_tokens=True)
            outputs.append([{"output_text": out_txt, "output_tokens": beam, "orig_output_tokens": orig_beam.tolist(), "logprob": lp.item(), "score": score.item()}
                            for out_txt, beam, orig_beam, lp, score in zip(output_txts, beams, orig_beams, beam_logprobs, beam_scores)])

        if timings:
            print("outputs", time.time()-T)
            T = time.time()

        return outputs

    def generate(self, bodies, max_batch_size=8, beam_size=1, ckpt_runs=1, num_runs=1, progress=False, sort_score=False, keep_unique=False, **kwargs):
        # This function batches the generation and adds functionality for `num_runs` (running k independent runs of the same input), and dispatches it to the correct generation method:
        # `generate_beam_batch` if beam_size>1 (requires beam_size)
        # `generate_ckpt_batch` if ckpt_runs>1 (requires scorer, ckpt_runs, ckpt_every)
        # `generate_batch` otherwise

        assert not (beam_size > 1 and ckpt_runs > 1), "Cannot ask for beam search and ckpt generation at the same time"
        if ckpt_runs > 1:
            assert "ckpt_every" in kwargs and "scorer" in kwargs, "Required parameters were not fed to the generation function."

        N_start = len(bodies)
        if num_runs > 1:
            bodies = [body for body in bodies for i in range(num_runs)]
        N = len(bodies)

        outputs = []
        iterator = range(0, N, max_batch_size)
        if progress:
            iterator = tqdm.tqdm(iterator)
        for i in iterator:
            batch_bodies = bodies[i:min(N, i+max_batch_size)]
            with torch.no_grad():
                if beam_size > 1:
                    # print("Will run generate beam batch")
                    batch_outputs = self.generate_beam_batch(batch_bodies, beam_size=beam_size, **kwargs)
                elif ckpt_runs > 1:
                    # print("Will run generate ckpt batch")
                    batch_outputs = self.generate_ckpt_batch(batch_bodies, ckpt_runs=ckpt_runs, **kwargs)
                else:
                    # print("Will run generate batch")
                    batch_outputs = self.generate_batch(batch_bodies, **kwargs)
            outputs += batch_outputs

        if num_runs > 1:
            # Refold the number of runs into N outputs
            final_outputs = []
            for i in range(N_start):
                all_runs = outputs[num_runs*i:(num_runs*(i+1))]
                if beam_size > 1:
                    all_runs = [beam for beams in all_runs for beam in beams] # Unfold
                if sort_score:
                    sort_key = "score" if "score" in all_runs[0] else "logprob"
                    all_runs = sorted(all_runs, key=lambda o: -o[sort_key])
                if keep_unique:
                    already_outputs = set([])
                    unique_runs = []
                    for run in all_runs:
                        if run["output_text"] not in already_outputs:
                            unique_runs.append(run)
                            already_outputs.add(run["output_text"])
                    all_runs = unique_runs

                final_outputs.append(all_runs)
            outputs = final_outputs
        return outputs


if __name__ == "__main__":
    # import difflib, os
    import utils_misc

    MODELS_FOLDER = os.environ["MODELS_FOLDER"]
    utils_misc.select_freer_gpu()

    ################### TESTING GPT2 SIMPLIFIER #####################
    # model = Generator("gpt2-medium", max_output_length=90, device='cuda')
    # model.reload(os.path.join(MODELS_FOLDER, "simplifier/gen_mediumc_lamb_1.bin"))

    # text = "The revamped MoMA will not be a single narrative of one history. Rather, it will be a collection of perspectives, according to Ann Temkin. She is the museum's chief curator of painting and sculpture."

    # print("Original")
    # print(text)
    # print("========================")

    # generated_texts = model.generate([text], beam_size=1, num_runs=10)[0]

    # for generated_beam in generated_texts:
    #     print("==================")
    #     print("[%.3f]\n%s" % (generated_beam['score'], generated_beam['output_text']))

    ################### TESTING BART QGEN #####################
    # model = Generator("facebook/bart-large", max_output_length=90, device="cuda", seq2seq=True)
    # model.reload("/mnt/results/gen_qgen_bart_logprob_1.993.bin")

    # outputs = model.generate([text], beam_size=3, num_runs=3, sample=False, sort_score=True) # , force_start="How is"
    # for beams in outputs:
    #     print( "====================")
    #     for beam in beams:
    #         print("[%.3f]\n%s" % (beam['score'], beam['output_text']))

    ################### TESTING COPIER #####################
    # model = Generator("facebook/bart-base", max_output_length=90, device="cuda", seq2seq=True)
    # model = Generator("gpt2-medium", max_output_length=90, device="cuda")
    # model.reload("/home/phillab/models/gpt2_med_lede2.bin")
    # # model.reload("/home/phillab/models/gen_gpt2_med_cp90_logprob_0.001.bin")
    # # model.reload("/home/phillab/models/gpt2_med_cp90.bin")
    # model.model.half().eval()
    # text = "Romanian villagers have re-elected their mayor by a landslide even though he died two weeks ago from Covid-19 complications. They said he had done a good job and deserved his posthumous victory.
    # And the also added something else which I forgot for now."
    # with torch.no_grad():
    #     outputs = model.generate([text], num_runs=32, sample=True, no_copy_ngram=7, sort_score=True) # , force_start="How is"

    # for beams in outputs:
    #     print( "====================")
    #     for beam in beams:
    #         print("[%.3f]\n%s" % (beam['logprob'], beam['output_text']))

    ################### TESTING QGEN #####################
    # text = "With most concerts, events, and international travel still off limits for Americans, national and state parks have seen a dramatic uptick in visitors over the few months leading up to September.
    #       For example, Yellowstone National Park saw a 7.5 percent increase in August visitors compared to 2019, making it the second-busiest August in park history. Other parks have seen similar increases."

    # model = Generator("gpt2-medium", max_output_length=20, device="cuda")
    # model.reload("/home/phillab/models/qgen/gpt2_med_newsqa_only_logprob_2.059.bin")

    # for force_start in ["Who", "What", "Why", "How", "When"]:
    #     outputs = model.generate([text], beam_size=2, num_runs=3, sample=False, sort_score=True, force_start=force_start)
    #     for beams in outputs:
    #         print( "====================")
    #         for beam in beams:
    #             print("%s" % (beam['output_text']))

    ################### KEEP IT SIMPLE: ABLATION MODELS #####################
    ori_paragraph = """A small capsule containing asteroid soil samples that was dropped from 136,700 miles in space by Japan's Hayabusa2 spacecraft landed as planned in the Australian Outback on December 6.
                    The extremely high precision required to carry out the mission thrilled many in Japan, who said they took pride in its success."""

    from model_guardrails import SimplificationRShift, SimplificationVShift # HallucinationPenalty
    from model_discriminator import FluencyRelativeScore
    from model_coverage import CoverageModel
    from utils_masking import KeywordMasker
    import utils_edits

    vshift_target, word_change_ratio = 0.4, 0.15

    coverage_model_file = os.path.join(MODELS_FOLDER, "coverage_roberta_kw30p.bin")
    coverage_scorer = CoverageModel(KeywordMasker(mask_ratio=0.4), model_file=coverage_model_file, fp16=True, is_soft=True)

    scorers = [{"name": "vshift", "model": SimplificationVShift(target_shift=vshift_target, word_change_ratio=word_change_ratio), "sign": 1, "weight": 2.0},
               {"name": "rshift", "model": SimplificationRShift(), "sign": 1, "weight": 2.0},
               {"name": "fluency", "model": FluencyRelativeScore(), "sign": 1},
               {"name": "coverage", "model": coverage_scorer, "sign": 1, "weight": 3.0}
               ]

    scorer = utils_scoring.ScorerWrapper(scorers, scoring_method="logsum", max_batch_size=12)

    model = Generator("gpt2-medium", max_output_length=90, device="cuda")
    model.reload("/home/phillab/models/simplifier/gen_mediumc_lamb_1.bin")
    model.model.half().eval()

    preds = model.generate([ori_paragraph], num_runs=8, sample=True, sort_score=True, scorer=scorer)[0]

    outs = [p["output_text"] for p in preds]
    scores = scorer([ori_paragraph] * len(outs), outs)

    for i, pred in enumerate(preds):
        print("-----------")
        print("[[ %.3f ]]" % (pred["logprob"]))
        print(" ".join(["%s: %.3f" % (k, scores[k][i]) for k in scores.keys()]))
        print(utils_edits.show_diff_word(ori_paragraph, pred["output_text"]))
        print("---")
        print(utils_edits.show_diff_word(ori_paragraph, pred["output_text"], is_latex=True))

    ################### TESTING PEGASUS/ARXIV #####################
    # documents = open("/home/phillab/data/arxiv/validation.source", "r").readlines()

    # # {"max_input_length": 1024, "max_output_length": 256, "min_length": 32, "no_repeat_ngram": 0}
    # model = Generator("google/pegasus-arxiv", seq2seq=True, max_input_length=1024)
    # model.model.eval()

    # chosen_doc = [documents[8]]

    # for max_output_length in [89,90,91]:
    # # for max_output_length in range(90, 130, 1):
    #     print("=====================")
    #     # print("========= %d ===========" % (max_output_length))
    #     # print("=====================")
    #     model_output = model.generate(chosen_doc, beam_size=4, max_output_length=max_output_length, min_length=min(max_output_length, 32), no_repeat_ngram=0)[0]

    #     nl_model_output = torch.LongTensor([beam["orig_output_tokens"] for beam in model_output]).cuda()
    #     nl_model_output = nl_model_output[torch.argsort(torch.sum(nl_model_output, dim=-1))] # Reorder by something sensible
    #     print("[NL %d]" %(max_output_length), nl_model_output[:, -10:].cpu().numpy())

    #     # HF
    #     input_ids = model.tokenizer(chosen_doc, truncation=True, max_length=1024, return_tensors="pt")
    #     input_ids = {k: v.cuda() for k, v in input_ids.items()}
    #     hf_model_output = model.model.generate(**input_ids, max_length=max_output_length, num_beams=4, num_return_sequences=4) # , num_beams=args.beam_size, max_length=10
    #     hf_model_output = hf_model_output[torch.argsort(torch.sum(hf_model_output, dim=-1))] # Reorder by something sensible

    #     print("[HF %d]\n" % (max_output_length), hf_model_output[:, -10:].cpu().numpy())

        # print(torch.all(hf_model_output == nl_model_output))
        # print(hf_model_output.shape, nl_model_output.shape)

        # print("----------------------------")
        # for beam in model_output:
        #     print("[%.3f] %s" % (beam["logprob"], beam["output_text"]))

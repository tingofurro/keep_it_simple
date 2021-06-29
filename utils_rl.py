import time, numpy as np, torch
from datetime import datetime
from apex import amp
import utils_edits

def select_logprobs(logits, decoded_tokens, eos_id):
    logprobs = torch.nn.functional.log_softmax(logits, dim=2)

    selected_logprobs = []
    for i, generated_tokenized in enumerate(decoded_tokens):
        generated_tokenized.append(eos_id)
        generated_tokenized = generated_tokenized[:generated_tokenized.index(eos_id)] # Remove probs of stuff after end tokens
        selected_logprob = logprobs[i, torch.arange(len(generated_tokenized)), generated_tokenized]
        summed_logprob = torch.sum(selected_logprob)
        selected_logprobs.append(summed_logprob)
    selected_logprobs = torch.stack(selected_logprobs, dim=0)
    return selected_logprobs

class ReinforceCriterion:
    def __init__(self, generator, optimizer, use_apex=False):
        self.generator = generator
        self.optimizer = optimizer
        self.eos_id = self.generator.tokenizer.eos_token_id
        self.use_apex = use_apex

    def __call__(self, encoded_inputs, decoded_tokens, rewards):
        # Note: rewards should already be an advantage (reward - baseline)

        assert len(encoded_inputs)==len(decoded_tokens), "There's a shape mismatch between inputs and outputs %d != %d" % (len(encoded_inputs), len(decoded_tokens))

        logits = self.generator.train_batch(encoded_inputs, decoded_tokenized=decoded_tokens, return_logits=True)
        selected_logprobs = select_logprobs(logits, decoded_tokens, self.eos_id)

        loss = torch.mean(rewards * selected_logprobs)
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

class RLThermostat:
    def __init__(self):
        self.temperature = 1.0
        self.threshold_enough = 0.7
        self.step = 0.1

    def log_diversity(self, diversity):
        if diversity <= self.threshold_enough:
            # Increase temperature, there was not enough diversity
            self.temperature += self.step
        elif self.temperature > 1.0:
            self.temperature -= self.step
        return self.temperature

class RLModelCheckpoint:
    def __init__(self, model, ckpt_every, ckpt_lookback, ckpt_file):
        self.model = model
        self.ckpt_every = ckpt_every
        self.ckpt_lookback = ckpt_lookback
        self.best_ckpt_score = None
        self.score_history = []
        self.ckpt_file = ckpt_file
        self.time_start = time.time()
        self.time_ckpt = time.time()

    def tick(self, latest_score):
        self.score_history.append(latest_score)
        is_this_best = False
        if time.time() - self.time_start > 30*60 and len(self.score_history) > self.ckpt_lookback:
            # Don't do anything for the first 30 minutes
            current_score = np.mean(self.score_history[-self.ckpt_lookback:])

            if time.time()-self.time_ckpt > self.ckpt_every:
                revert_ckpt = self.best_ckpt_score is not None and current_score < min(1.15*self.best_ckpt_score, 0.85*self.best_ckpt_score) # Could be negative or positive
                print("================================== CKPT "+datetime.now().strftime("%Y-%m-%d %H:%M:%S")+" =================================")
                print("[CKPT] Previous best: %.4f vs. current: %.4f" % ((0.0 if self.best_ckpt_score is None else self.best_ckpt_score), current_score))
                print("[CKPT] Am I reverting? %s" % ("yes" if revert_ckpt else "no! BEST CKPT"))

                if revert_ckpt:
                    self.model.model.load_state_dict(torch.load(self.ckpt_file))
                self.time_ckpt = time.time()
                print("============================== END OF CKPT TIME ==============================")

            is_this_best = self.best_ckpt_score is None or current_score > self.best_ckpt_score
            if is_this_best:
                print("[CKPT] Saved new best at: %.4f" % (current_score))
                self.best_ckpt_score = current_score
                torch.save(self.model.model.state_dict(), self.ckpt_file)
        return is_this_best

class RLExamplePrinter:
    def __init__(self, print_every, N_samples, print_source=False, print_edit=False):
        self.print_every = print_every
        self.N_samples = N_samples
        self.print_source = print_source
        self.print_edit = print_edit
        self.time_print = time.time()

    def tick(self, paragraphs, generateds, scorer_returns):
        if time.time()-self.time_print > self.print_every:
            IDX = int(np.argmax(scorer_returns['total_scores']) / self.N_samples)
            if self.print_source:
                print("----------- ORIGINAL -------------")
                print(paragraphs[IDX])

            print("----------- GENERATED OPTIONS ---------")
            gen_is = sorted(range(self.N_samples*IDX, self.N_samples*(IDX+1)), key=lambda gen_i: -scorer_returns["total_scores"][gen_i]) # Ordered from best scoring to smallest scoring

            for gen_i in gen_is:
                if self.print_edit:
                    print(utils_edits.show_diff_word(paragraphs[IDX], generateds[gen_i]))
                else:
                    print(generateds[gen_i])
                print("["+"; ".join(["%s: %.4f"% (k.replace("_scores", ""), scorer_returns[k][gen_i]) for k in scorer_returns if ("_score" in k or "pred_level" in k)])+"]")
                print("---")

            self.time_print = time.time()
            print("==========================================")

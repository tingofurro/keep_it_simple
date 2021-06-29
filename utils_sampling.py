import torch.nn.functional as F
import torch

def _top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
    r"""Adapted from
    https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317#file-top-k-top-p-py-L16-L27"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep also the first token above the
    # threshold
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    for idx in range(logits.size(0)):
        batch_indices = sorted_indices[idx, sorted_indices_to_remove[idx]]
        logits[idx, batch_indices] = float("-inf")
    return logits

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1): # float("Inf")
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """

    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        # print(indices_to_remove)
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        # print(indices_to_remove)

        logits[indices_to_remove] = filter_value

    return logits

def ngram_copy_filtering(generateds, no_copy_texts, logits, n_gram=3):
    # `generateds`: tensor (N, L_1) a list of sequences of generated tokens which cannot repeat more than `n_gram` tokens from the no_copy_text
    # `no_copy_texts`: tensor (N,L_2) a list of sequences of tokens which will be used to limit the logits so that generateds cannot repeat more than n_gram subsequences from this text
    # `logits`: (N,vocab_size) if the generateds would have the possibility to form a (n_gram+1) copy-ngram from no_copy_texts, then make those tokens very unlikely
    if n_gram <= 0 or generateds is None:
        return logits

    N1, L1 = generateds.shape
    N2, L2 = no_copy_texts.shape
    assert N1 == N2, "The number of elements in generateds and no_copy_texts do not match (%d != %d)" % (N1, N2)

    if L1 < n_gram or L2 < n_gram:
        return logits

    generateds = generateds.tolist()
    no_copy_texts = no_copy_texts.tolist()

    for i, (generated, no_cp_txt) in enumerate(zip(generateds, no_copy_texts)):
        last_k = generated[-n_gram:]
        start_idxs = [x for x in range(L2-n_gram) if no_cp_txt[x:(x+n_gram)] == last_k]
        to_remove = [no_cp_txt[start_idx+n_gram] for start_idx in start_idxs]
        if len(to_remove)>0:
            logits[i, to_remove] -= 1000.0
    return logits

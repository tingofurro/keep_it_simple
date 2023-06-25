from tqdm import tqdm


def pre_process_min_max_fluency_log_probs(fluency_model, dataloader):
    min_value = None
    max_value = None

    for sources in tqdm(dataloader):
        sources_score: float = fluency_model.text2loss(sources).cpu().tolist()[0]

        if min_value is None:
            min_value = sources_score
        elif sources_score < min_value:
            min_value = sources_score

        if max_value is None:
            max_value = sources_score
        elif sources_score > max_value:
            max_value = sources_score
    return min_value, max_value

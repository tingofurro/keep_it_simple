import string
from collections import Counter

import pandas as pd

data = pd.read_csv(
    "./results/run_coverage_masking_analysis_k_8_glo-supercomputer_components_steps_values.csv"
)

counter = Counter()

for idx, row in data.iterrows():
    cleaned_texts = [text.strip() for text in row[1].split(";")]
    counter.update(cleaned_texts)

print(counter)
# '' mean a real double quote `"`, since we clean it during the transformation into a list.
most_common = counter.most_common(200)
n = sum(counter.values())
print(most_common)
ratio_most_common = [
    (word, round(frequency / n * 100, 2)) for word, frequency in most_common
]
print(ratio_most_common)

punctuations = string.punctuation + "``—''"
punctuations_ratios = [
    (word, ratio) for word, ratio in ratio_most_common if word in punctuations
]
print(
    f"Punctuations case {round(sum([ratio for _, ratio in punctuations_ratios]), 2)}%:",
)

words_junction = ["'s", "nt"]
words_junction_ratios = [
    (word, ratio) for word, ratio in ratio_most_common if word in words_junction
]
data_punctuation = punctuations_ratios
data_punctuation.extend(words_junction_ratios)
print(
    f"Punctuations case and words junction {round(sum([ratio for _, ratio in data_punctuation]), 2)}%:",
)

number_ratios = []
for word, ratio in ratio_most_common:
    try:
        float(word)
        number_ratios.append((word, ratio))
    except:
        pass
print(
    f"Number masked {round(sum([ratio for _, ratio in number_ratios]), 2)}%:",
)
print(
    "The mean coverage ratio for this experimentation is:",
    data["coverage_effective_mask_ratio"].mean(),
)
print(
    "The mean coverage score for this experimentation is:",
    data["mean_coverage_scores"].mean(),
)

data = pd.read_csv(
    "./results/run_coverage_masking_analysis_no_context_glo-supercomputer_components_steps_values.csv"
)

print(
    "The mean coverage ratio for this experimentation (no context) is:",
    data["coverage_effective_mask_ratio"].mean(),
)

print(
    "The mean coverage ratio score for the no context is:",
    data["mean_coverage_scores"].mean(),
)

data = pd.read_csv(
    "./results/run_coverage_masking_analysis_no_punctuations_glo-supercomputer_components_steps_values.csv"
)

print(
    "The mean coverage ratio for this experimentation (no punctuations) is:",
    data["coverage_effective_mask_ratio"].mean(),
)

print(
    "The mean coverage ratio score for the no punctuations is:",
    data["mean_coverage_scores"].mean(),
)

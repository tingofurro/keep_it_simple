# # level base on FK and use by https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8988250 and
# # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8771811/#supplemental-information
# flesh_kincaid_level = [
#     "Very_Easy",
#     "Easy",
#     "Fairly_Easy",
#     "Plain_English",
#     "Fairly_Difficult",
#     "Difficult",
#     "Very_Difficult",
# ]
#
# mapping_newsela_to_level = {
#     "12-11": "Fairly_Difficult",
#     "10-8": "Plain_English",
#     "7-5": "Fairly_Easy",
#     "4-2": "Very_Easy",
# }
#
# mapping_FCCLC_to_level = {}

import pandas as pd
from text_complexity_computer import TextComplexityComputer
from tqdm import tqdm

seed = 42
n_iter = 10000
cv = 3
verbose = 1
n_cores = 12

prescaled = False

root = "./"

metadata = pd.read_csv("../datastore/articles_metadata.csv")
data = []

tcc = TextComplexityComputer(scaler=None, language="en")

for row_metadata in tqdm(metadata[0:10].iterrows(), total=len(metadata)):
    row_metadata_serie = row_metadata[1]
    if row_metadata_serie["language"] == "en":
        Y = row_metadata_serie["grade_level"]
        file_name = row_metadata_serie["filename"]
        with open(f"../datastore/articles/{file_name}", "r", encoding="utf8") as file:
            all_text = file.readlines()
            paragraphs = "".join(all_text).split("\n\n")
            for paragraph_idx, paragraph in enumerate(paragraphs):
                X = tcc.get_metrics_scores(paragraph)
                data.append(X.iloc[0, :].tolist())

pd.DataFrame(data).to_csv("process_newsela_data.csv", index=False)

import os.path

import pandas as pd
from text_complexity_computer import TextComplexityComputer
from tqdm import tqdm

root = "../"

metadata = pd.read_csv(os.path.join(root, "datastore", "articles_metadata.csv"))
data = []

tcc = TextComplexityComputer(scaler=None, language="en")

for row_metadata in tqdm(metadata.iterrows(), total=len(metadata)):
    row_metadata_serie = row_metadata[1]
    if row_metadata_serie["language"] == "en":
        Y = row_metadata_serie["grade_level"]
        file_name = row_metadata_serie["filename"]
        with open(f"../datastore/articles/{file_name}", "r", encoding="utf8") as file:
            all_text = file.readlines()
            paragraphs = "".join(all_text).split(
                "\n\n"
            )  # We split paragraphs to the double newline character
            for paragraph_idx, paragraph in enumerate(paragraphs):
                X = tcc.get_metrics_scores(paragraph)
                data_to_append = [Y]
                data_to_append.extend(X.iloc[0, :].tolist())
                data.append(data_to_append)

columns = ["Y"]
columns.extend(X.columns)
pd.DataFrame(data, columns=columns).to_csv(
    os.path.join(root, "datastore", "pre_process_newsela_data.csv"), index=False
)

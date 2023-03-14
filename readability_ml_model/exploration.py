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
import os

import pandas as pd

seed = 42
n_iter = 10000
cv = 3
verbose = 1
n_cores = 12

root = "./"

all_data = pd.read_csv(os.path.join(root, "datastore", "process_newsela_data.csv"))

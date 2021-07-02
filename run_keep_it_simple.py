import utils_misc, argparse, os, utils_edits

utils_misc.select_freer_gpu()
from model_generator import Generator
# from model_access_simplifier import AccessSimplifier

MODELS_FOLDER = os.environ["MODELS_FOLDER"]

parser = argparse.ArgumentParser()
parser.add_argument("--model_card", type=str, default="gpt2-medium", help="Either `gpt2` or `gpt2-medium`")
parser.add_argument("--model_file", type=str, required=True, help="Use for example `gpt2_med_keep_it_simple.bin` provided in the codebase.")

args = parser.parse_args()

model = Generator(args.model_card, max_output_length=90, device='cuda')

if len(args.model_file) > 0:
    model.reload(args.model_file)
model.eval()

paragraph = """NASA's Curiosity rover just celebrated a major milestone — 3,000 days on the surface of Mars.
               To mark the occasion, the space agency has released a stunning new panorama of the red planet, captured by the rover."""

model_output = model.generate([paragraph], num_runs=8, sample=True)[0]

print("ORIGINAL TEXT:")
print(paragraph)

print(utils_edits.show_diff_word("Legend: Deletions are in red. Additions are in.", "Legend: Deletions are in. Additions are in green."))

print(model_output)

for candidate in model_output:
    print("----")
    print(utils_edits.show_diff_word(paragraph, candidate["output_text"]))

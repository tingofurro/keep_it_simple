import Levenshtein
from colorama import Fore
import nltk, numpy as np

def fix_punctuation(text):
    return text.replace(" ,", ",").replace(" .", ".").replace(" n't", "n't").replace(" 's", "'s").replace(" 'm", "'m").replace(" 're", "'re")

# At the Character-level, most likely not useful.
def show_diff(original, modified):
    """
    http://stackoverflow.com/a/788780
    Unify operations between two compared strings seqm is a difflib.
    SequenceMatcher instance whose a & b are strings
    """
    # seqm = difflib.SequenceMatcher(None, modified, original)
    output= []
    opcodes = Levenshtein.opcodes(modified, original)
    for opcode, a0, a1, b0, b1 in opcodes:
        if opcode == 'equal':
            output.append(modified[a0:a1])
        elif opcode == 'insert':
            output.append(Fore.RED + original[b0:b1] + Fore.RESET)
        elif opcode == 'delete':
            output.append(Fore.GREEN + modified[a0:a1] + Fore.RESET)
        elif opcode == 'replace':
            # seqm.a[a0:a1] -> seqm.b[b0:b1]
            # print(seqm.)
            output.append(Fore.RED + original[b0:b1] + Fore.RESET + Fore.GREEN + modified[a0:a1] + Fore.RESET)

    return ''.join(output)

# Given 2 texts, build Levenshtein MxN matrix
def build_levenshtein_matrix(text1, text2):
    r = nltk.tokenize.word_tokenize(text1)
    h = nltk.tokenize.word_tokenize(text2)

    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint16)
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    return d, r, h

# Convert the Levenshtein matrix to a sequence of colors Red Green Black colors
def word_sequence_color(d, r, h):
    word_sequence = []
    x, y = len(r), len(h)

    while x > 0 and y > 0:
        if r[x - 1] == h[y - 1]:
            x = x - 1
            y = y - 1
            word_sequence.append({"w": h[y], "color": "black", "idx1": x, "idx2": y})
        elif d[x][y] == d[x - 1][y - 1] + 1:    # substitution
            x = x - 1
            y = y - 1
            word_sequence.append({"w": h[y], "color": "green", "idx1": x, "idx2": y})
            word_sequence.append({"w": r[x], "color": "red", "idx1": x, "idx2": y})
        elif d[x][y] == d[x - 1][y] + 1:        # deletion
            x = x - 1
            word_sequence.append({"w": r[x], "color": "red", "idx1": x, "idx2": y})
        elif d[x][y] == d[x][y - 1] + 1:        # insertion
            y = y - 1
            word_sequence.append({"w": h[y], "color": "green", "idx1": x, "idx2": y})
    while x > 0:
        x = x - 1
        word_sequence.append({"w": r[x], "color": "red", "idx1": x, "idx2": y})
    while y > 0:
        y = y - 1
        word_sequence.append({"w": h[y], "color": "green", "idx1": x, "idx2": y})
    return word_sequence[::-1]

# Used in the compute_span_edits function
def build_up2edit(build_up):
    greens = [w for w in build_up if w["color"] == "green"]
    reds = [w for w in build_up if w["color"] == "red"]

    idx1 = min([w["idx1"] for w in build_up])
    to_words = lambda ws: fix_punctuation(" ".join([w["w"] for w in ws]))

    if len(greens) > 0 and len(reds) > 0:
        return {"type": "substitution", "insertion": to_words(greens), "deletion": to_words(reds), "idx": idx1, "end_idx": idx1 + len(reds)}
    elif len(greens) > 0:
        return {"type": "insertion", "insertion": to_words(greens), "idx": idx1, "end_idx": idx1} # +1 For now removing the +1 seeing what happens
    elif len(reds) > 0:
        return {"type": "deletion", "deletion": to_words(reds), "idx": idx1, "end_idx": idx1+len(reds)}

# Compute word sequence color, and build up span edits
def compute_span_edits(text1, text2):
    d, r, h = build_levenshtein_matrix(text1, text2)
    word_sequence = word_sequence_color(d, r, h)

    span_edits = []
    build_up = []
    for w in word_sequence:
        if w["color"] == "black" and len(build_up) > 0:
            span_edits.append(build_up2edit(build_up))
            build_up = []
        elif w["color"] != "black":
            build_up.append(w)

    if len(build_up) > 0:
        span_edits.append(build_up2edit(build_up))
        build_up = []
    return span_edits

# Printer function in either BASH or HTML
def show_diff_word(text1, text2, is_HTML=False, is_latex=False, print_red=True, print_green=True):
    lefts = {"green": Fore.GREEN, "red": Fore.RED, "black": ""}
    rights = {"green": Fore.RESET, "red": Fore.RESET, "black": ""}
    if is_HTML:
        lefts["green"], lefts["red"] = "<span class='green'>", "<span class='red'>"
        rights["green"], rights["red"] = "</span>", "</span>"
    if is_latex:
        lefts["green"], lefts["red"] = "{\color{ForestGreen}", "{\color{red}"
        rights["green"], rights["red"] = "}", "}"

    edits = compute_span_edits(text1, text2)
    idx2edit = {edit["idx"]: edit for edit in edits}

    w1s = nltk.tokenize.word_tokenize(text1)
    i = 0
    final_text = []
    while i < len(w1s):
        if i in idx2edit:
            edit = idx2edit[i]
            if len(edit.get("deletion", "")) > 0 and print_red:
                final_text.append(lefts["red"]+edit["deletion"]+rights["red"])
            if len(edit.get("insertion", "")) > 0 and print_green:
                final_text.append(lefts["green"]+edit["insertion"]+rights["green"])
            del idx2edit[i] # Delete it so we don't loop on insertions, only apply it once.
            i = edit["end_idx"]
        else:
            final_text.append(w1s[i])
            i += 1
    return fix_punctuation(" ".join(final_text))
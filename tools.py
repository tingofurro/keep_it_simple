import argparse


def bool_parse(arg):
    if arg.lower() in ("true", "t", "yes", "y", "1"):
        return True
    elif arg.lower() in ("false", "f", "no", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

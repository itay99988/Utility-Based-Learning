"""
	This file contains the arguments to parse at command line.
"""
import argparse


def get_args():
    """
		Description:
		Parses arguments at command line.
		Parameters:
			None
		Return:
			args - the arguments parsed
	"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', dest='mode', type=str, default='lookahead')
    parser.add_argument('--experiment', dest='experiment', type=str, default='cross7')
    parser.add_argument('--repeat', dest='repeat', type=int, default=10)
    args = parser.parse_args()

    return args

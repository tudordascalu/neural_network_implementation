import numpy as np
import matplotlib.pyplot as plt
import sys


arguments = sys.argv
has_arguments = len(arguments) > 1

if not has_arguments or "1" in arguments:
    """
    Train neural network and save model to file
    """
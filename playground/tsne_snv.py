# vim: fdm=indent
'''
author:     Fabio Zanini
date:       29/11/18
content:    
'''
# Modules
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['SINGLET_CONFIG_FILENAME'] = 'singlet.yml'
sys.path.append('/home/fabio/university/postdoc/singlet')
from singlet.dataset import Dataset


# Script
if __name__ == '__main__':

    ds = Dataset(
            counts_table='dengue',
            samplesheet='dengue',
            )


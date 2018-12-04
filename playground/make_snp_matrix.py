# vim: fdm=indent
'''
author:     Fabio Zanini
date:       03/12/18
content:    Try recalling the frequencies from the VCF files.
'''
import os
import pysam
from collections import Counter


if __name__ == '__main__':

    # Go through all the files
    n_lines = {}
    n_lines_hist = Counter()
    fdn = '../bigdata/DENV_singleCellVCF'
    for fn in os.listdir(fdn):
        with pysam.VariantFile('{:}/{:}'.format(fdn, fn), 'r') as f:
            nl = sum(1 for line in f)
        n_lines[fn] = nl
        n_lines_hist[nl] += 1




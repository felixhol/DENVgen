# vim: fdm=indent
'''
author:     Fabio Zanini
date:       03/12/18
content:    Try recalling the frequencies from the VCF files.
'''
import os
import sys
import pysam
import numpy as np
import pandas as pd
import xarray as xr
from collections import Counter, defaultdict
from Bio import SeqIO

os.environ['SINGLET_CONFIG_FILENAME'] = 'singlet.yml'
sys.path.append('/home/fabio/university/postdoc/singlet')
from singlet.dataset import Dataset


if __name__ == '__main__':

    ds = Dataset(
            samplesheet='dengue',
        )

    ## Histogram of SNVs
    #n_lines = {}
    #n_lines_hist = Counter()
    #fdn = '../bigdata/DENV_singleCellVCF'
    #for fn in os.listdir(fdn):
    #    with pysam.VariantFile('{:}/{:}'.format(fdn, fn), 'r') as f:
    #        nl = sum(1 for line in f)
    #    n_lines[fn] = nl
    #    n_lines_hist[nl] += 1

    # Example file
    fdn = '../bigdata/DENV_singleCellVCF'
    fn_ex = 'vars1001700612_I2.vcf'
    f = pysam.VariantFile('{:}/{:}'.format(fdn, fn_ex), 'r')
    poss = defaultdict(list)
    for line in f:
        poss[line.pos].append(line)

    # Prepare output data
    seq = SeqIO.read('../data/U87411_strain16681.gb', 'gb')
    samplenames = ds.samplenames.tolist()
    positions = np.arange(len(seq))
    alpha = np.array(['A', 'C', 'G', 'T', '-', 'N'])
    alphal = alpha.tolist()
    afs = np.zeros((len(alpha), len(positions), len(samplenames)))
    cov = np.zeros((len(positions), len(samplenames)), int)

    # Default to the reference
    for ipos, a in enumerate(seq):
        ia = alphal.index(a)
        afs[ia, ipos, :] = 1

    # Fill the output
    fdn = '../bigdata/DENV_singleCellVCF'
    for fn in os.listdir(fdn):
        sn = fn_ex[4:-4]
        isn = samplenames.index(sn)
        with pysam.VariantFile('{:}/{:}'.format(fdn, fn), 'r') as f:
            for line in f:
                ipos = line.pos - 1
                cov[ipos, isn] = line.info['DP']
                if len(line.alts) == 1:
                    aff = [line.info['AF']]
                else:
                    aff = line.info['AF']
                iaa = alphal.index(line.ref)
                for a, af in zip(line.alts, aff):
                    ia = alphal.index(a)
                    afs[ia, ipos, isn] = af
                    afs[iaa, ipos, isn] -= af

    afs = xr.DataArray(
        data=afs,
        dims=['allele', 'position', 'sample'],
        coords={'allele': alpha, 'position': positions, 'sample': samplenames},
        )
    cov = xr.DataArray(
        data=cov,
        dims=['position', 'sample'],
        coords={'position': positions, 'sample': samplenames},
        )
    data = xr.Dataset(data_vars={'afs': afs, 'depth': cov, 'ref': np.array(seq)})

    # NetCDF is large but fast to I/O
    data.to_netcdf(path='../bigdata/allele_frequencies.nc', format='NETCDF4')
    # To open back the allele frequencies:
    # import xarray as xr
    # a = xr.open_dataarray('../data/allele_frequencies.nc')

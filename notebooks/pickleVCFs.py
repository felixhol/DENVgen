from pysam import VariantFile
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

baseDir = "/scratch/users/fhol/elife_data/"
saveDir = "/scratch/users/fhol/elife_data/varfilesDENV01/"
dataDir = glob.glob(baseDir+'/10017006*')

for d in dataDir:
    filename = glob.glob(d+"/*.vcf")
    for i in filename:
        df = pd.DataFrame(columns = ['pos', 'af'])
        varFileName = os.path.basename(i)
        SNVs = VariantFile(i)
        for rec in SNVs.fetch():
            df2 = pd.DataFrame([[rec.pos, rec.info["AF"]]], columns = ['pos', 'af'])
            df = df.append(df2, ignore_index = True)
        os.chdir(saveDir)
        df.to_pickle(os.path.splitext(varFileName)[0] +'_df.pkl')
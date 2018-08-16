#### to run demultiplex
module load system singularity
singularity run quakelab-singularity-bcl2fastq.img --output-dir Unaligned_L1 --sample-sheet 180803_SPENSER_0369_000000000-AY50H_L1_rci5_samplesheet.csv --barcode-mismatches 1 --use-bases-mask 1:y251,I8,I8,y251 --ignore- missing-bcls --ignore-missing-positions --ignore-missing-filter
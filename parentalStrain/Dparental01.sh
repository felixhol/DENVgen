#!/bin/bash
#echo "Script executed from: ${PWD}"
module load biology bowtie2
module load biology samtools
for i in /scratch/users/fhol/DENVparental/AY50H/DENV_amp1
	do cd "$i"
        date
	echo "Mapping: ${PWD}" 
#	ls *1.fastq.gz
	bowtie2 --local -x ../DENV2 -1 *R1_001.fastq.gz -2 *R2_001.fastq.gz -S "${PWD##*/}"BW_Sam.sam
	samtools sort -o "${PWD##*/}"_BW.sorted.bam *BW_Sam.sam
	samtools index *_BW.sorted.bam
#	/home/users/fhol/lofreq/lofreq_star-2.1.3.1/bin/lofreq call -f ../DENV2_test.fasta -o vars${PWD##*/}.vcf *_BW.sorted.bam
#	samtools idxstats *_BW.sorted.bam > idxstats${PWD##*/}.tsv
	date
        cd ../
done

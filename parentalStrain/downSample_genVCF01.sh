#!/bin/bash
### downsample mapped Sam and generate VCFs
for i in /scratch/users/fhol/DENVparental/AY50H/DENV_amp*/
	do cd "$i"
	module load biology samtools
	samtools view -h -s 0.025 *BW_Sam.sam > ${PWD##*/}_down025.bam
	samtools sort -o ${PWD##*/}_down025.sorted.bam *_down025.bam
	samtools index *_down025.sorted.bam
	/home/users/fhol/lofreq/lofreq_star-2.1.3.1/bin/lofreq call -f ../DENV2_test.fasta -o vars${PWD##*/}_d025.vcf *down025.sorted.bam
	samtools idxstats *down025.sorted.bam > idxstats${PWD##*/}.tsv
	date
	cd ../
done
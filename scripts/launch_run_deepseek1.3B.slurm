#!/bin/bash
#SBATCH -J cpt-v0-deepseek-1.3B           # Job name
#SBATCH -o slurm-outputs/cpt-deepseek-1.3B.o%j       # Name of stdout output file
#SBATCH -e slurm-outputs/cpt-deepseek-1.3B.e%j       # Name of stderr output file
#SBATCH -p gh          # Queue (partition) name
#SBATCH -N 1
#SBATCH -t 24:00:00        # Run time (hh:mm:ss)
#SBATCH -A AST24021       # Allocation name (req'd if you have more than 1)

srun ${WORK}/LLaMA-Factory/scripts/run_deepseek_cpt_1.3B.sh $SLURM_JOB_NODELIST $SLURM_JOB_NUM_NODES
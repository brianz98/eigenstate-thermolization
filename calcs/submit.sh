#!/bin/bash
#SBATCH -J et
#SBATCH -p LONG
#SBATCH -t 48:00:00
#SBATCH -C old
#SBATCH -n 1 -N 1 -c 4
#SBATCH --mail-user=zz376@cantab.ac.uk
#SBATCH --mail-type=FAIL

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
module load gcc/7.5.0 anaconda/python3/2022.05 mkl/64/2022/0/0
cp energy_disorder.py ${SLURM_JOB_ID}.py
python3 ${SLURM_JOB_ID}.py > ${SLURM_JOB_ID}.out 

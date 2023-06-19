#!/bin/bash
#SBATCH --mem=16gb
#SBATCH -c2
#SBATCH --time=1:00:0
#SBATCH -o JobsOutput/Job_STDOUT_%j.txt  # send stdout to outfile
#SBATCH -e JobsOutput/Job_STDERR_%j.txt  # send stderr to errfile
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=mor.turgeman2@mail.huji.ac.il
#SBATCH --gres=gpu:1,vmem:10g

python3 main.py -1 -1


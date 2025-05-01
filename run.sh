#!/bin/bash -l

#$ -P ds340       # Specify the SCC project name you want to use
#$ -l h_rt=12:00:00   # Specify the hard time limit for the job
#$ -N model_creation           # Give job a name
#$ -j y               # Merge the error and output streams into a single file


. venv/bin/activate
python final.py
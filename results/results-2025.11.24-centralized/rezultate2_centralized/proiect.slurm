#!/bin/bash
#SBATCH --job-name=aitdm    # Job name
#SBATCH -A student                              # SLURM account to be used for this job
#SBATCH --output=results_output_%j.txt              # Output file
#SBATCH --error=results_error_%j.txt                # Error log file
#SBATCH --gres=gpu:2                            # Request 2 GPUs
#SBATCH --ntasks=1                                          # Number of tasks (processes)
#SBATCH --cpus-per-task=4                       # Number of CPU cores per task (adjust as needed)
#SBATCH --time=11:59:00                         # Time limit (format: hh:mm:ss)
#SBATCH --mem=48G                               # Memory per node (adjust as needed)
#SBATCH --partition=dgxh100

source /export/home/mecanica/stud/m/matei.neaga/miniconda3/bin/activate ro_auth_detect

export PYTHONPATH=$PYTHONPATH:/mnt/pachete:/export/home/mecanica/stud/m/matei.neaga/dizertatie/stylometrics

  python patchcore_training.py \
      --dataset-root data/raw \
      --output-dir artifacts/patchcore \
      --coreset-method random \
      --coreset-ratio 0.5 \
      --coreset-max-samples 100000 \
      --batch-size 8 \
      --num-workers 4 \
      --image-size 512 \
      --coreset-chunk-size 16384 \
      --distance-chunk-size 8192 \
      --log-level INFO

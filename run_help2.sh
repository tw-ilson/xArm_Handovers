sbatch --job-name=$1 --dependency=afternotok:$2 --export=ALL $3

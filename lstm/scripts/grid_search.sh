#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --requeue
#SBATCH --job-name=hypSearch
#SBATCH --partition=gpu
#SBATCH --account=mx6
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ebd@njit.edu

module load wulver foss/2024a Python/3.12.3 CUDA/12.6.0
module li
echo 'Loading environment'
cd /mmfs1/project/mx6/ebd/SAR_EMERGENCE_RESEARCH/
source sar/bin/activate
pip -V
echo 'Done loading environment'

python -u lstm/grid_search.py $1

EOT
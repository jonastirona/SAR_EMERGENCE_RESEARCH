#PBS -S /usr/bin/sh
#PBS -N no_schd_transformer

## On NAS, model can be sky_gpu, cas_gpu or mil_a100 which have GPUs
## cas_gpu node: 24/48 cores 384GB, 4 GPUs (each 32GB)
## sky_gpu node: 18/36 cores 384GB, 4 GPUs (each 32GB)
## mil_a100 node: 64 cores / host, 16 / vnode, 256 GB / host, 64 / vnode, 4 GPUs / host, 1 / vnode (each 80GB)

#PBS -l select=1:model=mil_a100:ncpus=16:ngpus=1:mem=48GB
#PBS -l place=scatter:excl

#PBS -q p_gpu_normal
#PBS -l walltime=23:59:00

## PBS will email when the job is aborted, begun, ended.
#PBS -m abe
#PBS -M vishal.gaur@uah.edu

## join the stderr output to stdout, by default the output file will be placed in place where
## the qsub is run.  It can be in a different place with PBS -o /path/to/pbs/log/file
## by default, the output filename is jobname.oJOBID

#PBS -kod -ked
#PBS -o pbs_logs/transformer_no_sched.out    
#PBS -e pbs_logs/transformer_no_sched.err

NUM_NODES=1
TOTAL_NUM_GPUs=$((NUM_NODES * 1))  # Total number of gpus over all nodes: NUM_NODES * ngpus

export BASE=$PWD
export MASTER_PORT=19410
export MASTER_ADDR=$(hostname -i)
export WORLD_SIZE=$TOTAL_NUM_GPUs
export NODE_RANK=0
JOB_ID=$PBS_JOBID

NODES=($(uniq $PBS_NODEFILE))
echo cluster nodes: ${NODES[@]}

#FINAL_OUTPUT_FILE="pbs_logs/${JOB_ID}.out"

if [[ "$NUM_NODES" -ne ${#NODES[@]} ]]; then
    echo "Aborting, NUM_NODES and nodes requested are not consistent"
    exit 2
fi

#if [ -f pbs_logs/output.temp ]; then
#    mv pbs_logs/output.temp "$FINAL_OUTPUT_FILE"
#fi

# for each node that's not the current node
C=1
for node in ${NODES[@]}
do
  if [[ $node != $(hostname) ]]
  then
    # ssh into each node and run the .sh script with node info
    # run in background
    ssh -o StrictHostKeyChecking=no -i $HOME/.ssh/id_rsa $node "cd $BASE; sh shell_scripts/run.sh $C $NUM_NODES $WORLD_SIZE $MASTER_ADDR $MASTER_PORT $JOB_ID $PBS_JOBID $TMPDIR" &
    C=$((C + 1))
    sleep 2
  fi
done

# process on master node runs the last!

module purge
module use -a /swbuild/analytix/tools/modulefiles
module load miniconda3/v4
export CONDA_ENVS_PATH=/nobackupnfs1/sroy14/.conda/envs
export CONDA_PKGS_DIRS=/nobackupnfs1/sroy14/.conda/pkgs
source activate heliofm

echo 'current directory is'$PWD
export PYTHONPATH=$PWD
echo 'Number of nodes '$NUM_NODES
echo 'World Size '$WORLD_SIZE
echo 'Rendezvous address '$RDZV_ADDR
echo 'Rendezvous port '$RDZV_PORT
echo 'Rendezvous ID '$RDZV_ID
echo 'job ID '$PBS_JOBID


python train_w_stats.py 12 4 110 3 64 1000 0.01 st_transformer

echo "Done with PBS" 
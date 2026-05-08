export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

JOB_NAME=''
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
PARTITION=''
NNODE=1
NUM_GPUS=4
NUM_CPU=32

srun -p ${PARTITION} \
    -n${NNODE} \
    --gres=gpu:${NUM_GPUS} \
    --ntasks-per-node=1 \
    --cpus-per-task=${NUM_CPU} \
    bash torchrun.sh \
    --nnodes=${NNODE} \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:10066 \
    tasks/train_it.py \
    $(dirname $0)/config_7b_aot_only_ivs2.py \
    output_dir ${OUTPUT_DIR}

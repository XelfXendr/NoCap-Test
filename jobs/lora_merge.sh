#!/bin/bash
#PBS -q gpu@pbs-m1.metacentrum.cz
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb:gpu_mem=16gb:scratch_local=64gb:cluster=galdor
#PBS -l walltime=24:00:00
#PBS -N lora_merge

# storage is shared via NFSv4
HOMEDIR="/storage/brno2/home/$LOGNAME"
DATADIR="$HOMEDIR/NoCap-Test"
LOGDIR="$HOMEDIR/nocap_logs"

UV="$HOMEDIR/.local/bin/uv"
UVX="$HOMEDIR/.local/bin/uvx"

WANDB_KEY=$(cat $DATADIR/wandb_key)

# sets the scratchdir as a temporary dir, bypassing the metacentre disk quota
export TMPDIR=$SCRATCHDIR
# clean the SCRATCH when job finishes
trap 'clean_scratch' TERM EXIT

# copy working directory into the scratchdir
cp -r $DATADIR $SCRATCHDIR
cd $SCRATCHDIR/NoCap-Test

# set up python environment
$UV sync --cache-dir "$SCRATCHDIR/uv_cache"

# ... the computation ...
$UV run train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --model d12 \
  --batch_size 16 \
  --grad_accumulation_steps 32 \
  --sequence_length 1024 \
  --val_loss_every 128 \
  --val_batch_size 16 \
  --num_iterations 4768 \
  --weight_decay 0.1 \
  --learning_rate 0.0018 \
  --warmup_iters 256 \
  --warmdown_iters 1024 \
  --log_wandb \
  --wandb_key $WANDB_KEY \
  --lora_rank 8 \
  --lora_alpha 8.0 \
  --merge_every 100

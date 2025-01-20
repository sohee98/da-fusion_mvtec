#!/bin/bash
#SBATCH --job-name=f-tune
#SBATCH --exclude=matrix-1-12,matrix-0-24,matrix-1-4,matrix-2-13,matrix-1-8,matrix-0-38,matrix-1-18,matrix-1-20
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --partition=russ_reserved
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32g
#SBATCH --array=0-39
 
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate semantic-aug
# cd ~/spurge/semantic-aug

# RANK=$SLURM_ARRAY_TASK_ID WORLD_SIZE=$SLURM_ARRAY_TASK_COUNT \
# CUDA_VISIBLE_DEVICES=1 \
# python fine_tune.py --dataset=coco --output_dir=./output_coco_2 \
# --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
# --resolution=512 --train_batch_size=4 --lr_warmup_steps=0 \
# --gradient_accumulation_steps=1 --max_train_steps=1000 \
# --learning_rate=5.0e-04 --scale_lr --lr_scheduler="constant" \
# --mixed_precision=fp16 --revision=fp16 --gradient_checkpointing \
# --only_save_embeds --num-trials 1 --examples-per-class 4

CUDA_VISIBLE_DEVICES=1 \
python train_classifier.py --logdir coco-baselines/textual-inversion-0.5 \
--synthetic-dir "output_coco/aug/textual-inversion-0.5/{dataset}-{seed}-{examples_per_class}" \
--dataset coco --prompt "a photo of a {name}" \
--aug textual-inversion --guidance-scale 7.5 \
--strength 0.5 --mask 0 --inverted 0 \
--num-synthetic 10 --synthetic-probability 0.5 \
--num-trials 1 --examples-per-class 4

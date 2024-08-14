#!/bin/bash -l
#SBATCH --job-name=LFM-Superres
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=0-12:00:00
#SBATCH --mail-user=benearnthof@hotmail.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=LFM-Superres.out

source /dss/dsshome1/lxc01/ru25jan4/miniconda3/bin/activate
conda activate /dss/dsshome1/lxc01/ru25jan4/miniconda3/envs/imagen

python /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/LFM/LFM/downstream_tasks/train_flow_latent_inpainting.py \
  --exp superres_kl --dataset celeba \
  --batch_size 64 --lr 5e-5 --scale_factor 0.18215 --num_epoch 500 --image_size 128 \
  --num_in_channels 8 --num_out_channels 4 --ch_mult 1 2 3 4 --attn_resolution 16 8 \
  --num_process_per_node 2 --save_content

#!/bin/bash
# run_experiments.sh - Main training script with automatic authentication

# Stop script execution if any command fails
set -e

# =============================================
# TRAINING CONFIGURATION
# =============================================
WANDB_ENTITY="${WANDB_ENTITY:-refined-gae}"
WANDB_PROJECT="${WANDB_PROJECT:-Refined-GAE}"
HF_USERNAME="${HF_USERNAME}"
HF_REPO_ID="${HF_REPO_ID:-batmangiaicuuthegioi/refined-gae-checkpoints}"  # Default repo nếu không set

echo ""
echo "📊 Training configuration:"
echo "  - Wandb Entity: $WANDB_ENTITY"
echo "  - Wandb Project: $WANDB_PROJECT"
echo "  - HuggingFace Repo: $HF_REPO_ID"
echo ""

# =============================================
# OGBL-DDI EXPERIMENTS
# =============================================
echo "======================================"
echo "    OGBL-DDI EXPERIMENTS"
echo "======================================"

# DDI Baseline (no feature, yes embed, residual in encoder & decoder)
# echo "===================="
# echo "RUNNING DDI BASELINE"
# echo "===================="
# python train_wo_feat.py \
#   --dataset ogbl-ddi --lr 0.001 --hidden 1024 --batch_size 8192 \
#   --dropout 0.6 --num_neg 1 --epochs 20 --prop_step 2 --metric hits@20 \
#   --residual 0.1 --maskinput --mlp_layers 8 --mlp_res --emb_dim 1024 \
#   --wandb_entity $WANDB_ENTITY --wandb_project $WANDB_PROJECT \
#   --wandb_run_name ddi-baseline --save_model --hf_repo_id $HF_REPO_ID

# # DDI Depth sweep 2 -> 3
# echo "============================"
# echo "RUNNING DDI DEPTH SWEEP 3"
# echo "============================"
# python train_wo_feat.py \
#   --dataset ogbl-ddi --lr 0.001 --hidden 1024 --batch_size 8192 \
#   --dropout 0.6 --num_neg 1 --epochs 20 --prop_step 3 --metric hits@20 \
#   --residual 0.1 --maskinput --mlp_layers 8 --mlp_res --emb_dim 1024 \
#   --wandb_entity $WANDB_ENTITY --wandb_project $WANDB_PROJECT \
#   --wandb_run_name ddi-depth-sweep-3 --save_model --hf_repo_id $HF_REPO_ID

# # DDI Residual in Encoder OFF
# echo "=================================="
# echo "RUNNING DDI RESIDUAL ENCODER OFF"
# echo "=================================="
# python train_wo_feat.py \
#   --dataset ogbl-ddi --lr 0.001 --hidden 1024 --batch_size 8192 \
#   --dropout 0.6 --num_neg 1 --epochs 20 --prop_step 2 --metric hits@20 \
#   --residual 0.0 --maskinput --mlp_layers 8 --mlp_res --emb_dim 1024 \
#   --wandb_entity $WANDB_ENTITY --wandb_project $WANDB_PROJECT \
#   --wandb_run_name ddi-residual-encoder-off --save_model --hf_repo_id $HF_REPO_ID

# # DDI Residual in Decoder OFF
# echo "=================================="
# echo "RUNNING DDI RESIDUAL DECODER OFF"
# echo "=================================="
# python train_wo_feat.py \
#   --dataset ogbl-ddi --lr 0.001 --hidden 1024 --batch_size 8192 \
#   --dropout 0.6 --num_neg 1 --epochs 20 --prop_step 2 --metric hits@20 \
#   --residual 0.1 --maskinput --mlp_layers 8 --emb_dim 1024 \
#   --wandb_entity $WANDB_ENTITY --wandb_project $WANDB_PROJECT \
#   --wandb_run_name ddi-residual-decoder-off --save_model --hf_repo_id $HF_REPO_ID

# =============================================
# OGBL-COLLAB EXPERIMENTS
# =============================================
echo ""
echo "======================================"
echo "    OGBL-COLLAB EXPERIMENTS"
echo "======================================"

# COLLAB Baseline (yes feature, no embed)
echo "======================"
echo "RUNNING COLLAB BASELINE"
echo "======================"
# python collab.py \
#   --dataset ogbl-collab --lr 0.0004 --emb_hidden 0 --hidden 1024 \
#   --batch_size 16384 --dropout 0.2 --num_neg 3 --epochs 20 --prop_step 4 \
#   --metric hits@50 --mlp_layers 5 --res --norm --dp4norm 0.2 --scale \
#   --wandb_entity $WANDB_ENTITY --wandb_project $WANDB_PROJECT \
#   --wandb_run_name collab-baseline --save_model --hf_repo_id $HF_REPO_ID

# # COLLAB Depth sweep 4 -> 5
echo "=============================="
echo "RUNNING COLLAB DEPTH SWEEP 5"
echo "=============================="
python collab.py \
  --dataset ogbl-collab --lr 0.0004 --emb_hidden 0 --hidden 1024 \
  --batch_size 16384 --dropout 0.2 --num_neg 3 --epochs 20 --prop_step 5 \
  --metric hits@50 --mlp_layers 5 --res --norm --dp4norm 0.2 --scale \
  --wandb_entity $WANDB_ENTITY --wandb_project $WANDB_PROJECT \
  --wandb_run_name collab-depth-sweep-5 --save_model --hf_repo_id $HF_REPO_ID

# COLLAB Raw feature and embed
echo "====================================="
echo "RUNNING COLLAB RAW FEATURE AND EMBED"
echo "====================================="
python collab.py \
  --dataset ogbl-collab --lr 0.0004 --emb_hidden 1024 --hidden 1024 \
  --batch_size 16384 --dropout 0.2 --num_neg 3 --epochs 20 --prop_step 4 \
  --metric hits@50 --mlp_layers 5 --res --norm --dp4norm 0.2 --scale \
  --wandb_entity $WANDB_ENTITY --wandb_project $WANDB_PROJECT \
  --wandb_run_name collab-raw-feat-and-embed --save_model --hf_repo_id $HF_REPO_ID

# COLLAB Embed, no raw feature
echo "=================================="
echo "RUNNING COLLAB EMBED NO RAW FEATURE"
echo "=================================="
python train_wo_feat.py \
  --dataset ogbl-collab --lr 0.0004 --hidden 1024 --batch_size 16384 \
  --dropout 0.2 --num_neg 3 --epochs 20 --prop_step 4 --metric hits@50 \
  --residual 0.1 --maskinput --mlp_layers 5 --mlp_res --emb_dim 1024 \
  --wandb_entity $WANDB_ENTITY --wandb_project $WANDB_PROJECT \
  --wandb_run_name collab-embed-no-raw-feat --save_model --hf_repo_id $HF_REPO_ID

echo ""
echo "======================================"
echo "  ✅ ALL EXPERIMENTS COMPLETED!"
echo "======================================"

#!/bin/bash
# setup_auth.sh - Tự động cấu hình authentication cho Wandb và HuggingFace

# Màu sắc để dễ nhìn
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  SETUP AUTHENTICATION SCRIPT${NC}"
echo -e "${GREEN}========================================${NC}"

# =====================
# WANDB SETUP
# =====================
echo -e "\n${YELLOW}[1/2] Setting up Wandb...${NC}"

# Kiểm tra xem WANDB_API_KEY đã được set chưa
if [ -z "$WANDB_API_KEY" ]; then
    echo -e "${RED}❌ WANDB_API_KEY environment variable not found!${NC}"
    echo "Please set it by running:"
    echo "  export WANDB_API_KEY='your_wandb_api_key_here'"
    exit 1
else
    echo -e "${GREEN}✅ WANDB_API_KEY found in environment${NC}"
    # Login wandb bằng API key
    wandb login "$WANDB_API_KEY" --relogin
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Wandb login successful!${NC}"
    else
        echo -e "${RED}❌ Wandb login failed!${NC}"
        exit 1
    fi
fi

# =====================
# HUGGINGFACE SETUP
# =====================
echo -e "\n${YELLOW}[2/2] Setting up HuggingFace...${NC}"

# Kiểm tra xem HF_TOKEN đã được set chưa
if [ -z "$HF_TOKEN" ]; then
    echo -e "${RED}❌ HF_TOKEN environment variable not found!${NC}"
    echo "Please set it by running:"
    echo "  export HF_TOKEN='your_huggingface_token_here'"
    exit 1
else
    echo -e "${GREEN}✅ HF_TOKEN found in environment${NC}"
    # Login huggingface bằng token
    git config --global credential.helper store
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ HuggingFace login successful!${NC}"
    else
        echo -e "${RED}❌ HuggingFace login failed!${NC}"
        exit 1
    fi
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  ✅ ALL AUTHENTICATION COMPLETED!${NC}"
echo -e "${GREEN}========================================${NC}"

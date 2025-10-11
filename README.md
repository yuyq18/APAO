# Codes for APAO

## 0. Environment Setup

1. Create a new Conda environment:

```bash
conda create -n apao python=3.12.11
```

2. Activate the environment:

```bash
conda activate apao
```

3. Install [Pytorch](https://pytorch.org/get-started/locally/) and other required dependencies via `pip`:

```bash
# Take CUDA 12.6 as an example, you can change it to your desired version
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
# Install other dependencies
pip install -r requirements.txt
```
**Note**: Ensure that the version of GCC/G++ is >= 9.0.0.

## 1. Training with APAO

Run the scripts

```bash
bash scripts/train_apao_pointwise.sh
bash scripts/train_apao_pairwise.sh
```

Example Command (from `scripts/train_apao_pointwise.sh`):

```bash
export PYTHONPATH=./src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=3

export SEED=42

DATASET=Office  # Office, Grocery, Beauty, Yelp
method_name=T5_APAO_Pointwise  # T5_APAO_Pointwise, Llama_APAO_Pointwise

beta=0.1
prefix_tau=1.0

export WANDB_PROJECT="${DATASET}_${method_name}"
export WANDB_RUN_NAME="${method_name}_seed${SEED}_beta${beta}_prefixtau${prefix_tau}"

OUTPUT_DIR=./ckpt/${DATASET}_${method_name}/${WANDB_RUN_NAME}


python src/train/train_apao.py \
    --output_dir $OUTPUT_DIR \
    --seed $SEED \
    --dataset $DATASET \
    --wandb_run_name $WANDB_RUN_NAME \
    --method_name $method_name \
    --beta $beta \
    --prefix_tau $prefix_tau
```

## 2. Evaluation with APAO

Run the script

```bash
bash scripts/test_apao_pointwise.sh
bash scripts/test_apao_pairwise.sh
```

Example Command (from `scripts/test_apao_pointwise.sh`):

```bash
export PYTHONPATH=./src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=7

export SEED=42
beam_size=20
test_bsz=512

DATASET=Office  # Office, Grocery, Beauty, Yelp
method_name=T5_APAO_Pointwise  # T5_APAO_Pointwise, Llama_APAO_Pointwise

beta=0.1
prefix_tau=1.0

export WANDB_RUN_NAME="${method_name}_seed${SEED}_beta${beta}_prefixtau${prefix_tau}"

RESULTS_FILE=./results/${DATASET}_${method_name}/$WANDB_RUN_NAME/
CKPT_PATH=./ckpt/${DATASET}_${method_name}/${WANDB_RUN_NAME}/


python src/test/test_apao.py \
    --ckpt_path $CKPT_PATH \
    --seed $SEED \
    --dataset $DATASET \
    --method_name $method_name \
    --results_file $RESULTS_FILE \
    --test_batch_size $test_bsz \
    --num_beams $beam_size \
    --is_constrained_beam_search
```

## 3. Training with Baselines
Run the scripts

```bash
bash scripts/train_ce.sh
bash scripts/train_msl.sh
```

## 4. Evaluation with Baselines
Run the scripts

```bash
bash scripts/test_ce.sh
bash scripts/test_msl.sh
```
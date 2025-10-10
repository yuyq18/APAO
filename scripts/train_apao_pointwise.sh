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

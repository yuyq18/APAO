export PYTHONPATH=./src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=3

export SEED=42

DATASET=Office  # Office, Grocery, Beauty, Yelp
method_name=T5_CE  # T5_CE, Llama_CE

export WANDB_PROJECT="${DATASET}_${method_name}"
export WANDB_RUN_NAME="${method_name}_seed${SEED}"

OUTPUT_DIR=./ckpt/${DATASET}_${method_name}/${WANDB_RUN_NAME}


python src/train/train_ce.py \
    --output_dir $OUTPUT_DIR \
    --seed $SEED \
    --dataset $DATASET \
    --wandb_run_name $WANDB_RUN_NAME \
    --method_name $method_name

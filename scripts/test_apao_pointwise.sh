export PYTHONPATH=./src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

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

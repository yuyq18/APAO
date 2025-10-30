
import torch
import argparse
import os
from transformers import EarlyStoppingCallback
from transformers import TrainingArguments
from tokenizer.LightCodeTokenizer import LightCodeTokenizer

from transformers import T5Config, LlamaConfig
from utils.custom_trainer import CustomTrainer

from models import *
from utils.utils import *
from utils.collator import *
from utils.dataset import *
import wandb


def train(args):
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    device = torch.device("cuda", 0)

    # ==================================== Backbone model & Tokenizer ====================================
    print(f"Backbone Model: {args.backbone_name}, Method: {args.method_name}")

    config_class = eval("{}{}".format(args.backbone_name, "Config"))
    assert config_class in [T5Config, LlamaConfig], "Only T5 and Llama are supported now."

    args.model_config_file = f"./config/models/{args.backbone_name.lower()}_config.json"
    print(f"Loading model config from {args.model_config_file}")
    config = config_class.from_json_file(args.model_config_file)
    tokenizer = LightCodeTokenizer.from_pretrained(
        os.path.join(args.tokenizer_save_dir, args.dataset),
        model_max_length=512,
    )
    if args.backbone_name == "Llama" and args.decoder_only_data_mode == "inter-level":
        tokenizer.padding_side = "left"

    # save config & tokenizer
    vocab_size = len(tokenizer)
    config.vocab_size = vocab_size
    tokenizer.save_pretrained(args.output_dir)
    config.save_pretrained(args.output_dir)
    
    # ==================================== Dataset & Collator ====================================

    train_data, valid_data = load_datasets(args)
    if args.backbone_name == "T5":
        collator = Collator(args, tokenizer, add_input_eos=True)
    elif args.backbone_name == "Llama" and args.decoder_only_data_mode == "inter-level":
        collator = Collator(args, tokenizer, add_input_eos=False) # Set add_input_eos=False for inter-level data
    else:
        raise NotImplementedError

    # ==================================== Method ====================================
    model = method_class(
        config,
        args
    )
    model.resize_token_embeddings(vocab_size)
    model.to(device)

    # ==================================== Trainer ====================================

    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=20)
    ]

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=TrainingArguments(
            seed=args.seed,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            fp16=args.fp16,
            bf16=args.bf16,
            logging_steps=args.logging_step,
            optim=args.optim,
            eval_strategy=args.save_and_eval_strategy,
            save_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_steps,
            save_steps=args.save_and_eval_steps,
            output_dir=args.output_dir,
            save_total_limit=2,
            load_best_model_at_end=True,
            remove_unused_columns = False,
            dataloader_num_workers = args.dataloader_num_workers,
            dataloader_persistent_workers = False,
            label_names=['labels'],
        ),
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=callbacks
    )
    model.config.use_cache = False

    if args.resume_from_checkpoint:
        last_checkpoint = None
        if os.path.isdir(args.output_dir):
            checkpoints = [
                os.path.join(args.output_dir, d)
                for d in os.listdir(args.output_dir)
                if d.startswith("checkpoint")
            ]
            if checkpoints:
                last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train(resume_from_checkpoint=None)

    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)


if __name__ == "__main__":
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--method_name', type=str, default='T5_CE', help='method name')

    init_args, init_extras = init_parser.parse_known_args()

    method_class = eval("{}".format(init_args.method_name))

    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)
    parser = method_class.parse_model_args(parser)
    args, extras = parser.parse_known_args()

    args.backbone_name = init_args.method_name.split('_')[0]
    args.method_name = init_args.method_name

    # wandb
    if os.environ.get("WANDB_PROJECT", "debug") == "debug":
        os.environ["WANDB_MODE"] = "disabled"
    else:
        os.environ["WANDB_MODE"] = "online"
        wandb.init(project=os.environ.get("WANDB_PROJECT", "debug"), config=args)
        wandb.run.name = args.wandb_run_name
    
    train(args)
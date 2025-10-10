
import argparse
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Config, LlamaConfig
from tokenizer.LightCodeTokenizer import LightCodeTokenizer

from models import *
from utils.utils import *
from utils.collator import *
from utils.dataset import *
from utils.generation_trie import Trie

from test_utils import get_topk_results, get_metrics_results


def test(args):
    set_seed(args.seed)

    device_map = {"": args.gpu_id}
    device = torch.device("cuda", args.gpu_id)

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

    config.vocab_size = len(tokenizer)

    # ==================================== Dataset & Collator ====================================
    if "pointwise" in args.method_name.lower():
        if args.backbone_name == "T5":
            test_data = load_test_dataset(args)
            collator = TestCollator(args, tokenizer, add_input_eos=True)
        elif args.backbone_name == "Llama" and args.decoder_only_data_mode == "inter-level":
            test_data = load_test_dataset(args)
            collator = TestCollator(args, tokenizer, add_input_eos=False) # Set add_input_eos=False for inter-level data
        else:
            raise NotImplementedError
    elif "pairwise" in args.method_name.lower():
        if args.backbone_name == "T5":
            test_data = load_test_dataset(args)
            collator = TestCollator(args, tokenizer, add_input_eos=True)
        elif args.backbone_name == "Llama" and args.decoder_only_data_mode == "inter-level":
            test_data = load_test_dataset(args)
            collator = TestCollator(args, tokenizer, add_input_eos=False)
        else:
            raise NotImplementedError
        
    all_items = test_data.get_all_items()
    candidate_trie = Trie(
        [
            [0] + tokenizer.encode(candidate)
            for candidate in all_items
        ]
    )
    if args.is_constrained_beam_search:
        if args.backbone_name == "T5":
            prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)
            max_new_tokens = 5
        elif args.backbone_name == "Llama":
            prefix_allowed_tokens = prefix_allowed_tokens_fn_for_decoderonly(candidate_trie)
            max_new_tokens = 4
    else:
        prefix_allowed_tokens = None


    model = method_class.from_pretrained(
        args.ckpt_path,
        low_cpu_mem_usage=True,
        device_map=device_map
    )

    # ==================================== Test ====================================
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
                             shuffle=False, num_workers=args.dataloader_num_workers, pin_memory=True)

    model.eval()

    metrics = args.metrics.split(",")
    with torch.no_grad():
        metrics_results = {}
        total = 0

        for step, batch in enumerate(tqdm(test_loader)):
            inputs = batch[0].to(device)
            targets = batch[1]
            total += len(targets)

            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                prefix_allowed_tokens_fn=prefix_allowed_tokens,
                num_beams=args.num_beams,
                num_return_sequences=args.num_beams,
                output_scores=True,
                return_dict_in_generate=True,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            if args.backbone_name == "Llama":
                output_ids = output["sequences"][:, -4:]
            else:
                output_ids = output["sequences"]
            scores = output["sequences_scores"]

            decoded_output = tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            input_codes = tokenizer.batch_decode(
                inputs["input_ids"], skip_special_tokens=True
            )

            topk_res, rec_results_df = get_topk_results(
                decoded_output, 
                scores, 
                targets=targets, 
                k=args.num_beams,
                input_codes=input_codes,
            )

            batch_metrics_res = get_metrics_results(topk_res, metrics)
            for m, res in batch_metrics_res.items():
                if m not in metrics_results:
                    metrics_results[m] = res
                else:
                    metrics_results[m] += res

        for m in metrics_results:
            metrics_results[m] = metrics_results[m] / total
    

    print("======================================================")
    print("  |  ".join([f"{m:>5}" for m in metrics_results]), end="")
    print()
    print(", ".join([f"{metrics_results[m]:>4.4f}" for m in metrics_results]), end="")
    print()
    print("======================================================")


    save_data={}
    save_data["final_results"] = metrics_results

    ensure_dir(args.results_file)
    with open(os.path.join(args.results_file, "final_beam_search_results.json"), "w") as f:
        json.dump(save_data, f, indent=4)


if __name__ == "__main__":
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--method_name', type=str, default='T5_APAO_Pointwise', help='method name')

    init_args, init_extras = init_parser.parse_known_args()

    method_class = eval("{}".format(init_args.method_name))

    parser = argparse.ArgumentParser(description="LLMRec_test")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args, extras = parser.parse_known_args()

    args.backbone_name = init_args.method_name.split('_')[0]
    args.method_name = init_args.method_name

    test(args)
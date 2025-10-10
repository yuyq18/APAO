import math
import pandas as pd
import re

def get_topk_results(predictions, scores, targets, k, input_codes):
    results = []
    rec_results_df = pd.DataFrame(columns=["input", "target", "rec_items", "rec_scores"])
    B = len(targets)
    predictions = [_.strip().replace(" ","") for _ in predictions]
    inputs = [_.strip().replace(" ","") for _ in input_codes]

    for b in range(B):
        batch_seqs = predictions[b * k: (b + 1) * k]
        batch_scores = scores[b * k: (b + 1) * k]
        
        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        target_item = targets[b]
        one_results = []
        sorted_rec_items = [item[0] for item in sorted_pairs]
        sorted_scores = [item[1].item() for item in sorted_pairs]
        for sorted_pred in sorted_pairs:
            if sorted_pred[0] == target_item:
                one_results.append(1)
            else:
                one_results.append(0)

        batch_inputs = inputs[b]
        batch_tokens = re.findall(r"<[^>]+>", batch_inputs)
        # 每4个一组进行分组
        input_items = ["".join(batch_tokens[i:i+4]) for i in range(0, len(batch_tokens), 4) ]

        results.append(one_results)
        rec_results_df = pd.concat([rec_results_df, pd.DataFrame({"input": [input_items], "target": [target_item], "rec_items": [sorted_rec_items], "rec_scores": [sorted_scores]})], ignore_index=True)

    return results, rec_results_df

def get_metrics_results(topk_results, metrics):
    res = {}
    for m in metrics:
        if m.lower().startswith("hit"):
            k = int(m.split("@")[1])
            res[m] = hit_k(topk_results, k)
        elif m.lower().startswith("ndcg"):
            k = int(m.split("@")[1])
            res[m] = ndcg_k(topk_results, k)
        else:
            raise NotImplementedError

    return res


def ndcg_k(topk_results, k):
    """
    Since we apply leave-one-out, each user only have one ground truth item, so the idcg would be 1.0
    """
    ndcg = 0.0
    for row in topk_results:
        res = row[:k]
        one_ndcg = 0.0
        for i in range(len(res)):
            one_ndcg += res[i] / math.log(i + 2, 2)
        ndcg += one_ndcg
    return ndcg


def hit_k(topk_results, k):
    hit = 0.0
    for row in topk_results:
        res = row[:k]
        if sum(res) > 0:
            hit += 1
    return hit


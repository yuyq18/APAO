import os
from torch.utils.data import Dataset

import json
import numpy as np
import torch

class BaseDataset(Dataset):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)

        self.max_his_len = args.max_his_len
        self.index_file = args.index_file

        self.new_tokens = None
        self.allowed_tokens = None
        self.all_items = None
        self.all_items_inter_counts = dict()


    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)

    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))
        return self.new_tokens

    def get_all_items(self):
        if self.all_items is not None:
            return self.all_items

        self.all_items = set()
        for index in self.indices.values():
            self.all_items.add("".join(index))

        return self.all_items
    
    def get_all_items_inter_counts(self):
        return self.all_items_inter_counts

    def get_all_items_v2(self):
        if self.all_items is not None:
            return self.all_items

        self.all_items = []
        for index in self.indices.values():
            self.all_items.append("".join(index))
        return self.all_items      

    def get_prefix_allowed_tokens_fn(self, tokenizer):
        if self.allowed_tokens is None:
            self.allowed_tokens = {}
            for index in self.indices.values():
                for i, token in enumerate(index):
                    token_id = tokenizer(token)["input_ids"][0]
                    if i not in self.allowed_tokens.keys():
                        self.allowed_tokens[i] = set()
                    self.allowed_tokens[i].add(token_id)
            self.allowed_tokens[len(self.allowed_tokens.keys())] = set([tokenizer.eos_token_id])
        sep = [0]

        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()
            reversed_sent = sentence[::-1]
            for i in range(len(reversed_sent)):
                if reversed_sent[i:i + len(sep)] == sep[::-1]:
                    return list(self.allowed_tokens[i])

        return prefix_allowed_tokens_fn

    def _process_data(self):
        raise NotImplementedError


class SeqRecDataset(BaseDataset):
        
    def __init__(self, args, mode="train", sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.sample_num = sample_num

        # load data
        self._load_data()
        self._remap_items()
        
        # load data
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.inter_data = self._process_valid_data()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
            for item in self.indices.values():
                self.all_items_inter_counts["".join(item)] = 0

    def _remap_items(self):
        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items


    def _process_train_data(self):
        inter_data = []
        for uid  in self.remapped_inters:
            items = self.remapped_inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()
                # one_data["user"] = uid
                one_data["item"] = items[i]
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                one_data["inters"] = "".join(history)
                inter_data.append(one_data)
                self.all_items_inter_counts[items[i]] += 1
        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            # print(sample_idx[:10])##################
            inter_data = np.array(inter_data)[sample_idx].tolist()
        return inter_data
    
    def _process_valid_data(self):
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-2]
            self.all_items_inter_counts[items[-2]] += 1
            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            one_data["inters"] = "".join(history)
            inter_data.append(one_data)
        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            # print(sample_idx[:10])##################
            inter_data = np.array(inter_data)[sample_idx].tolist()
        return inter_data

    def _process_test_data(self):
        inter_data = []
        for uid in self.remapped_inters:
            # if uid not in cold_user:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-1]
            history = items[:-1]
            self.all_items_inter_counts[items[-1]] += 1
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            one_data["inters"] = "".join(history)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            # print(sample_idx[:10])##################
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data
    
    def _process_test_data_ids(self):
        inter_data = []
        for uid in self.inters:
            # if uid not in cold_user:
            items = self.inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            one_data["inters"] = history
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            # print(sample_idx[:10])##################
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data       

    def __len__(self):
        return len(self.inter_data)

    def __getitem__(self, index):
        d = self.inter_data[index]
        return dict(input_ids=d["inters"], labels=d["item"])

class SeqRecDecoderOnlyDataset(BaseDataset): # user-level dataset for decoder-only model
    def __init__(self, args, mode="train", sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.sample_num = sample_num

        # load data
        self._load_data()
        self._remap_items()
        
        # load data
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.inter_data = self._process_valid_data()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        elif self.mode == 'test_ranking':
            self.inter_data = self._process_test_data_ids()
        else:
            raise NotImplementedError

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
            for item in self.indices.values():
                self.all_items_inter_counts["".join(item)] = 0

    def _remap_items(self):
        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items


    def _process_train_data(self):
        inter_data = []
        for uid  in self.remapped_inters:
            items = self.remapped_inters[uid][:-2] # 把items[-3]放在train data最后一个位置，不显式定义label了
            one_data = dict()
            # breakpoint()
            # 这里相对于encoder-decoder架构做了trade-off, max_his_len=100，把max_his_len之前的数据扔了，同时让max_his_len内受到训练的数据看到了更多history
            if self.max_his_len > 0:
                history = items[-self.max_his_len:] 
            else:
                history = items
            one_data["inters"] = "".join(history)
            inter_data.append(one_data)
            # breakpoint()
            for i in range(1, len(items)):
                self.all_items_inter_counts[items[i]] += 1
        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            # print(sample_idx[:10])##################
            inter_data = np.array(inter_data)[sample_idx].tolist()
        return inter_data
    
    def _process_valid_data(self):
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-2]
            self.all_items_inter_counts[items[-2]] += 1
            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            one_data["inters"] = "".join(history)
            inter_data.append(one_data)
        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            # print(sample_idx[:10])##################
            inter_data = np.array(inter_data)[sample_idx].tolist()
        return inter_data

    def _process_test_data(self):
        inter_data = []
        for uid in self.remapped_inters:
            # if uid not in cold_user:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-1]
            history = items[:-1]
            self.all_items_inter_counts[items[-1]] += 1
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            one_data["inters"] = "".join(history)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            # print(sample_idx[:10])##################
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data
    
    def _process_test_data_ids(self):
        inter_data = []
        for uid in self.inters:
            # if uid not in cold_user:
            items = self.inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            one_data["inters"] = history
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            # print(sample_idx[:10])##################
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data       

    def __len__(self):
        return len(self.inter_data)

    def __getitem__(self, index):
        data = self.inter_data[index]
        if "item" not in data:
            return dict(input_ids=data["inters"])
        else:
            return dict(input_ids=data["inters"], labels=data["item"])


class NegSampleOnEpochDataset(SeqRecDataset):

    def __init__(self, args, mode="train", sample_num=-1, sampling="uniform", neg_k=100, tokenizer=None):
        super().__init__(args, mode, sample_num)

        self.tokenizer = tokenizer

        self.all_items_texts = np.array(list(self.get_all_items()), dtype=object)
        self.n_items = len(self.all_items_texts)

        # 提前编码所有item
        if self.tokenizer is not None:
            self.all_items_input_ids = [self.tokenizer.encode(candidate) for candidate in self.all_items_texts]
            self.all_items_input_ids = torch.tensor(self.all_items_input_ids, dtype=torch.long)

        self.sampling = sampling
        self.neg_k = neg_k

        # 本地 RNG（不使用全局 np.random）
        self._rng = np.random.default_rng(42)

        self.N_inters = len(self.inter_data)
        # (n_inters, neg_k)
        self.inter_negs = self._rng.integers(
            low=0, high=self.n_items,
            size=(self.N_inters, self.neg_k),
            dtype=np.int64
        )


    def resample_negs(self, generator: torch.Generator = None, seed: int = None, curr_model=None):
        if generator is not None:
            # 从 torch.Generator 拿到一个确定性的 32-bit 种子
            local_seed = int(generator.initial_seed() % (2**32))
            self._rng = np.random.default_rng(local_seed)
        elif seed is not None:
            self._rng = np.random.default_rng(int(seed % (2**32)))
        # else: 不改 self._rng，允许外部连续调用产生连贯随机流

        if self.sampling == "uniform":
            # 向量化一次性采样，避免 python for 循环
            self.inter_negs = self._rng.integers(
                low=0, high=self.n_items,
                size=(self.N_inters, self.neg_k),
                dtype=np.int64
            )
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        d = self.inter_data[index]
        output_dict = {
            "input_ids": d["inters"],
            "labels": d["item"],
        }
        # 负采样
        if self.sampling == "uniform":
            assert self.inter_negs is not None
            neg_idx = self.inter_negs[index]
            neg_items = self.all_items_input_ids[neg_idx]
            output_dict["neg_input_ids"] = neg_items # (neg_k, L)
        else:
            raise NotImplementedError
        return output_dict


import json
import re
from transformers import PreTrainedTokenizer
from typing import List, Optional, Union
from transformers.tokenization_utils_base import AddedToken

class LightCodeTokenizer(PreTrainedTokenizer):
    vocab_files_names = {
        "vocab_file": "vocab.json",
    }
    def __init__(self, vocab_file="vocab.json",
                 eos_token="</s>", pad_token="<pad>", unk_token="<unk>", **kwargs):
        with open(vocab_file, encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

        super().__init__(
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            **kwargs
        )

    def _tokenize(self, text: str) -> List[str]:
        # 使用 regex 拆分形如 <a_1><b_2>
        return [tok for tok in re.split(r"(<[^>]+>)", text) if tok.strip()]

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        if token_ids_1 is None:
            return token_ids_0
        token_ids_1 = self._add_eos_if_not_present(token_ids_1)
        return token_ids_0 + token_ids_1

    def _add_eos_if_not_present(self, token_ids: List[int]) -> List[int]:
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            return token_ids
        return token_ids + [self.eos_token_id]

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            return super().get_special_tokens_mask(token_ids_0, token_ids_1, already_has_special_tokens=True)
        if token_ids_1 is None:
            return [0] * len(token_ids_0) + [1]
        return [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [0] * len(token_ids_0)
        return [0] * (len(token_ids_0) + len(token_ids_1))

    def get_vocab(self) -> dict:
        return self.vocab

    def save_vocabulary(self, save_directory, filename_prefix=None):
        import os
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)
        return (vocab_file,)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        # Step 1: 调用父类逻辑（完成 AddedToken 封装、special token 注册等）
        added = super()._add_tokens(new_tokens, special_tokens=special_tokens)

        # Step 2: 获取父类构建好的 self._added_tokens_encoder
        for token_str, token_id in self._added_tokens_encoder.items():
            if token_str not in self.vocab:
                self.vocab[token_str] = token_id
                self.ids_to_tokens[token_id] = token_str

        return added
    
    def __len__(self):
        return len(self.vocab)
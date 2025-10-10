from transformers import TrainerCallback
import torch

class ResampleNegsCallback(TrainerCallback):
    def __init__(self, train_dataset, valid_dataset, base_seed: int = 42, model=None):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.base_seed = base_seed
        self.model = model

    def on_epoch_begin(self, args, state, control, offset_seed=2025, **kwargs):
        # state.epoch 可能是 float，这里取整；首次可能为 None
        epoch = int(state.epoch) if state.epoch is not None else 0
        rank = int(getattr(args, "process_index", 0))
        base = args.seed if getattr(args, "seed", None) is not None else self.base_seed

        print(f"Resample negs for epoch {epoch}...")
        train_seed = base + epoch + rank
        valid_seed = base + epoch + rank + offset_seed

        train_generator = torch.Generator(device="cpu").manual_seed(train_seed)
        valid_generator = torch.Generator(device="cpu").manual_seed(valid_seed)

        if self.train_dataset.sampling == "uniform":
            self.train_dataset.resample_negs(generator=train_generator)
            self.valid_dataset.resample_negs(generator=valid_generator)
        elif self.train_dataset.sampling in ["candidate_item_hard", "beam_search_hard"]:
            self.train_dataset.resample_negs(generator=train_generator, curr_model=self.model)
            self.valid_dataset.resample_negs(generator=valid_generator, curr_model=self.model)

        return control
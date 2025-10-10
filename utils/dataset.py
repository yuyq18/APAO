from utils.data import SeqRecDataset, NegSampleOnEpochDataset, SeqRecDecoderOnlyDataset

def load_datasets(args):
    train_data = SeqRecDataset(args, mode="train", sample_num=args.train_sample_num)
    valid_data = SeqRecDataset(args, mode="valid", sample_num=args.eval_sample_num)

    return train_data, valid_data

def load_decoderonly_datasets(args, mode="inter-level"):
    if mode == "user-level":
        train_data = SeqRecDecoderOnlyDataset(args, mode="train", sample_num=args.train_sample_num)
        valid_data = SeqRecDecoderOnlyDataset(args, mode="valid", sample_num=args.eval_sample_num)
    elif mode == "inter-level":
        train_data = SeqRecDataset(args, mode="train", sample_num=args.train_sample_num)
        valid_data = SeqRecDataset(args, mode="valid", sample_num=args.eval_sample_num)
    return train_data, valid_data

def load_neg_sample_on_epoch_datasets(args, tokenizer):
    train_data = NegSampleOnEpochDataset(args, mode="train", sample_num=args.train_sample_num, sampling=args.sampling, neg_k=args.neg_k, tokenizer=tokenizer)
    valid_data = NegSampleOnEpochDataset(args, mode="valid", sample_num=args.eval_sample_num, sampling=args.sampling, neg_k=args.neg_k, tokenizer=tokenizer)
    return train_data, valid_data

def load_test_dataset(args):
    test_data = SeqRecDataset(args, mode="test", sample_num=args.sample_num)
    return test_data

def load_test_decoderonly_dataset(args, mode="inter-level"):
    if mode == "user-level":
        test_data = SeqRecDecoderOnlyDataset(args, mode="test", sample_num=args.sample_num)
    elif mode == "inter-level":
        test_data = SeqRecDataset(args, mode="test", sample_num=args.sample_num)
    return test_data
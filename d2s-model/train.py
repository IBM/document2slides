#!/usr/bin/env python
"""
    Training
"""
import json
from pathlib import Path

from lfqa_utils import *
from log_utils import init_logger, logger


def train_bart(args):
    init_logger(args.logfile)
    logger.info('--- train params ---')
    logger.info('lr: {}'.format(args.lr))
    logger.info('bs: {}'.format(args.num_gpus*4))
    logger.info('epochs: {}'.format(args.max_epochs))
    logger.info('ir type: {}'.format(args.ir_type))
    logger.info('BartKeyword')
    logger.info('saving to: {}/{}'.format(args.model_path, args.model_name))
    Path(args.model_path).mkdir(parents=True, exist_ok=True)

    class ArgumentsS2S():
        def __init__(self):
            self.batch_size = args.num_gpus*2  # assuming each gpu is 16GB, 32GB can use 4 bs
            self.backward_freq = 16
            self.max_length = 1024
            self.print_freq = 100
            self.model_save_name = args.model_path+'/'+args.model_name
            self.learning_rate = args.lr
            self.num_epochs = args.max_epochs

    s2s_args = ArgumentsS2S()

    slide_train_docs = json.load(open('{}/precomputed/train_{}{}.json'
                                      .format(args.cache_path, args.ir_type)))
    slide_valid_docs = json.load(open('{}/precomputed/val_{}{}.json'
                                      .format(args.cache_path, args.ir_type)))
    s2s_train_dset = SlideDatasetS2S(slide_train_docs)
    s2s_valid_dset = SlideDatasetS2S(slide_valid_docs, training=False)

    s2s_tokenizer, pre_model = make_qa_s2s_model(
        model_name="facebook/bart-large-cnn",
        from_file=None,
        device="cuda:0"
    )
    s2s_model = torch.nn.DataParallel(pre_model, device_ids=list(range(args.num_gpus)))

    train_qa_s2s(s2s_model, s2s_tokenizer, s2s_train_dset, s2s_valid_dset, s2s_args, logfile=args.logfile)

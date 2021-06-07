#!/usr/bin/env python
"""
    Main workflow
"""
import argparse

from ir import dense_ir,  idf_recall
from train import train_bart
from test import test_model, validate, only_rouge


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='pick a mode',dest='mode')

    # shared params
    parser.add_argument("-model_path", default='../models/')
    parser.add_argument("-cache_path", default='../cache/')
    parser.add_argument("-result_path", default='../results/')

    # params for caching IR
    parser_ir = subparsers.add_parser('ir', help='cache information retrieval documents')
    parser_ir.add_argument("-slide_json", default='../input/acl_slides_filter.json')
    parser_ir.add_argument("-paper_path", default='../input/sciduet_papers/')
    parser_ir.add_argument("-split_path", default="../input/split/")
    parser_ir.add_argument("-ir_model", default="../models/bert_mix_ir.pth")
    parser_ir.add_argument("-filter", type=str2bool, nargs='?', const=True, default=True)
    parser_ir.add_argument("-eval", type=str2bool, nargs='?', const=True, default=False)

    # params for training
    parser_train = subparsers.add_parser('train', help='train d2s model')
    parser_train.add_argument("-max_epochs", default=10, type=int)
    parser_train.add_argument("-num_gpus", default=2, type=int)
    parser_train.add_argument("-lr", default=1e-4, type=float)
    parser_train.add_argument("-model_name", type=str, required=True)
    parser_train.add_argument("-ir_type", type=str, choices=['filter', 'prefilter'], required=True)
    parser_train.add_argument("-logfile", default='log.log')

    # params for test
    parser_test = subparsers.add_parser('test', help='test d2s model')
    parser_test.add_argument("-s2s_model", default="../models/bart_keyword.pth")
    parser_test.add_argument("-file_name", type=str, default="")

    # params for validate
    parser_val = subparsers.add_parser('val', help='validate d2s model')
    parser_val.add_argument("-s2s_model", default="../models/bart_keyword.pth")
    parser_val.add_argument("-file_name", type=str, default="")

    # params for rouge
    parser_rouge = subparsers.add_parser('rouge', help='compute rouge score')
    parser_rouge.add_argument("-result_file", required=True)

    args = parser.parse_args()

    print(args)

    if args.mode == 'ir':
        if args.eval:
            idf_recall(args)
        else:
            dense_ir(args)
    elif args.mode == 'train':
        train_bart(args)
    elif args.mode == 'validate':
        validate(args)
    elif args.mode == 'test':
        test_model(args)
    elif args.mode == 'rouge':
        only_rouge(args.result_file)

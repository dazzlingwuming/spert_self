import argparse

from args import train_argparser, eval_argparser, predict_argparser
from config_reader import process_configs
from spert import input_reader
from spert.spert_predictor import SpertPredictor
from spert.spert_trainer import SpERTTrainer


def _train():
    arg_parser = train_argparser()
    process_configs(target=__train, arg_parser=arg_parser)


def __train(run_args):
    trainer = SpERTTrainer(run_args)
    trainer.train(train_path=run_args.train_path, valid_path=run_args.valid_path,
                  types_path=run_args.types_path, input_reader_cls=input_reader.JsonInputReader)


def _eval():
    arg_parser = eval_argparser()
    process_configs(target=__eval, arg_parser=arg_parser)


def __eval(run_args):
    trainer = SpERTTrainer(run_args)
    trainer.eval(dataset_path=run_args.dataset_path, types_path=run_args.types_path,
                 input_reader_cls=input_reader.JsonInputReader)


def _predict():
    arg_parser = predict_argparser()
    process_configs(target=__predict, arg_parser=arg_parser)


def __predict(run_args):
    trainer = SpERTTrainer(run_args)
    trainer.predict(dataset_path=run_args.dataset_path, types_path=run_args.types_path,
                    input_reader_cls=input_reader.JsonPredictionInputReader)

def _local_predict():
    arg_parser = predict_argparser()
    process_configs(target=__local_predict, arg_parser=arg_parser,mutil_process=False)

def __local_predict(run_args):
    model_predict = SpertPredictor(types_path = run_args.types_path,
                             model_type = run_args.model_type,
                             model_path = run_args.model_path,
                             cache_path = run_args.cache_path,
                             tokenizer_path = run_args.tokenizer_path,
                             max_span_size = run_args.max_span_size,
                             max_pairs = run_args.max_pairs,
                             prop_drop= run_args.prop_drop,
                             size_embedding = run_args.size_embedding,
                             rel_filter_threshold = run_args.rel_filter_threshold)
    #进行预测
    while True:
        text = input("请输入文本，以空格隔开")
        if text == '':
            break
        predictions = model_predict.predict(text)
        print(predictions)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('mode', type=str, help="Mode: 'train' or 'eval'")
    args, _ = arg_parser.parse_known_args()

    if args.mode == 'train':
        _train()
    elif args.mode == 'eval':
        _eval()
    elif args.mode == 'predict':
        _predict()
    elif args.mode == 'local_predict':
        _local_predict()
    else:
        raise Exception("Mode not in ['train', 'eval', 'predict'], e.g. 'python spert.py train ...'")

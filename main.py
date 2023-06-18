import os
import timeit
import logging
import wandb
import argparse
import numpy as np
import pandas as pd
from os.path import exists
from huggingface_hub import interpreter_login

from transformers import TrainingArguments, HfArgumentParser, AutoConfig, AutoTokenizer, \
    AutoModelForSequenceClassification, set_seed, EvalPrediction, Trainer, TextClassificationPipeline
from datasets import load_dataset, Dataset, DatasetDict
# from dataclasses import dataclass, field
from evaluate import load
from sklearn.model_selection import train_test_split

os.environ["TOKENIZERS_PARALLELISM"] = "false"
wandb.login(key='94ee7285d2d25226f2c969e28645475f9adffbce')
logger = logging.getLogger(__name__)


def preprocess_wrapper(tokenizer):
    def preprocess_function(data):
        result = tokenizer(data['sentence'], truncation=True, max_length=512)
        return result

    return preprocess_function


def compute_metrics_wrapper(metric):
    def compute_metrics(p: EvalPrediction):
        predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        predictions = np.argmax(predictions, axis=1)
        result = metric.compute(predictions=predictions, references=p.label_ids)
        return result

    return compute_metrics


def load_amazon(filename):
    df = pd.read_csv(filename)
    df_new = pd.DataFrame()
    df_new['sentence'] = df['question'] + ' [SEP] ' + df['product_description']
    df_new['label'] = df['label']
    df_new = df_new.sample(frac=1)
    df_new = df_new[df_new['sentence'].notna()]

    train, test = train_test_split(df_new, test_size=0.2, shuffle=False)
    train = train[train['sentence'].notna()]
    test = test[test['sentence'].notna()]
    train['idx'] = range(0, len(train))
    test['idx'] = range(0, len(test))

    return train, test


def load_headlines(filename):
    def edit_headline(row):
        headline = row['original']
        edit_word = row['edit']
        res = headline[:headline.index('<')] + edit_word + headline[headline.index('>') + 1:]
        return res

    splits = ['train', 'dev', 'test', 'train_funlines']

    for split in splits:
        df = pd.read_csv(f'{filename}/old_{split}.csv')
        df_new = pd.DataFrame()
        df_new['id'] = df['id']
        df_new['meanGrade'] = df['meanGrade']
        df_new['sentence'] = df.apply(edit_headline, axis=1)
        df_new['label'] = df.apply(lambda row: 1 if row['meanGrade'] >= 1 else 0, axis=1)
        df_new.to_csv(f'{filename}/{split}.csv', index=False)


def load_train_test_from_csv(filename, data_name):
    if data_name == 'amazon':
        return load_amazon(filename)

    elif data_name == 'headlines':
        return load_headlines(filename)


def load_dataset_from_csv(filename, data_name):
    folder_name = filename.removesuffix('.csv')
    train_path = f'{folder_name}/train.csv'
    test_path = f'{folder_name}/test.csv'

    if exists(train_path) and exists(test_path):
        train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    else:
        train, test = load_train_test_from_csv(filename, data_name)

    train = train.sample(frac=1).reset_index(drop=True)
    ds = DatasetDict()
    ds['train'] = Dataset.from_pandas(train)
    ds['test'] = Dataset.from_pandas(test)

    if not exists(folder_name):
        os.mkdir(folder_name)
    train.to_csv(train_path)
    test.to_csv(test_path)

    return ds


def get_predictions_accuracy(preds_path, test_path):
    if exists(preds_path) and exists(test_path):
        preds = pd.read_csv(preds_path)
        test = pd.read_csv(test_path)
    equals = (test['label'] == preds['label']).sum()
    return equals / len(preds)


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('n_seeds', type=int)
    parser.add_argument('n_train_samples', type=int)
    # parser.add_argument('n_validation_samples', type=int)
    parser.add_argument('n_test_samples', type=int)
    args = parser.parse_args()

    return args


def get_split_set(dataset, split_name, num_of_samples):
    split_dataset = dataset[split_name].select(list(range(num_of_samples))) \
        if num_of_samples != -1 \
        else dataset[split_name]

    return split_dataset


def write_predictions(preds, test_dataset, data_name):
    if not exists('output/preds'):
        os.mkdir('output/preds')

    preds_dict = pd.DataFrame.from_dict(preds)
    preds_dict['sentence'] = test_dataset['sentence']
    preds_dict['label'] = preds_dict['label'].apply(lambda s: s[-1])
    preds_dict.to_csv(f'output/preds/{data_name}_preds_dict.csv')

    with open(f'output/preds/{data_name}_predictions.txt', 'w') as f:
        for i in range(len(preds)):
            f.write(f'{test_dataset["sentence"][i]}###{preds[i]["label"][-1]}\n')


def train_n_predict(models, dataset_path, train_on):
    print('\n~~ BEGIN ~~\n')

    my_args = load_args()
    training_args = TrainingArguments(output_dir='output', save_strategy='no', report_to=["wandb"])

    init_dataset = load_dataset_from_csv(dataset_path, train_on)

    print('\n~~ START OF TRAIN ~~\n')
    for model_name in models:
        metric = load('accuracy')

        for seed in range(my_args.n_seeds):
            set_seed(seed)
            training_args.run_name = f'{model_name}_seed_{seed}_trained_on_{train_on}'

            config = AutoConfig.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

            dataset = init_dataset.map(preprocess_wrapper(tokenizer), batched=True)

            train_dataset = get_split_set(dataset, 'train', my_args.n_train_samples)
            # validation_dataset = get_split_set(dataset, 'validation', my_args.n_validation_samples)
            test_dataset = get_split_set(dataset, 'test', my_args.n_test_samples)

            print(f'\n~~ START TRAIN ON {model_name}, seed = {seed} ~~\n')
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                # eval_dataset=validation_dataset,
                compute_metrics=compute_metrics_wrapper(metric),
                tokenizer=tokenizer,
            )

            trainer.train()
            print('\n~~ END OF TRAIN ~~\n')
            trainer.save_model(f'models/model_{model_name}_trained_on_{train_on}')

            text_classifier = TextClassificationPipeline(model=model.to('cpu'), tokenizer=tokenizer)

            print('\n~~ START OF PREDICT ~~\n')
            outputs = text_classifier(test_dataset['sentence'], batch_size=1)
            print('\n~~ END OF PREDICT ~~\n')

            write_predictions(outputs, test_dataset, 'amazon')

            init_headlines_dataset = get_dataset_headlines()
            headlines_dataset = init_headlines_dataset.map(preprocess_wrapper(tokenizer), batched=True)
            headlines_train_dataset = get_split_set(headlines_dataset, 'train', my_args.n_train_samples)
            headlines_test_dataset = get_split_set(headlines_dataset, 'test', my_args.n_test_samples)

            print('\n~~ START OF HEADLINES PREDICT ~~\n')
            headlines_outputs = text_classifier(headlines_test_dataset['sentence'], batch_size=1)
            print('\n~~ END OF HEADLINES PREDICT ~~\n')
            write_predictions(headlines_outputs, headlines_test_dataset, 'headlines')



def get_dataset_headlines():
    ds = DatasetDict()
    for split in ['train', 'test']:
        df = pd.read_csv(f'datasets/headlines/{split}.csv')
        ds[split] = Dataset.from_pandas(df)
    return ds


if __name__ == '__main__':
    models = ['roberta-base']
    amazon_reviews_path = 'datasets/amazon_reviews.csv'
    train_n_predict(models, amazon_reviews_path, 'amazon')

    # preds_path = 'output/amazon_preds.csv'
    # test_path = 'output/amazon_test.csv'
    # accuracy = get_predictions_accuracy(preds_path, test_path)
    # print(f'accuracy = {accuracy}')

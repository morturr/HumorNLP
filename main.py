import timeit
import logging
import wandb
import argparse
import numpy as np
import pandas as pd

from transformers import TrainingArguments, HfArgumentParser, AutoConfig, AutoTokenizer, \
    AutoModelForSequenceClassification, set_seed, EvalPrediction, Trainer, TextClassificationPipeline
from datasets import load_dataset
# from dataclasses import dataclass, field
from evaluate import load

wandb.login(key='94ee7285d2d25226f2c969e28645475f9adffbce')
logger = logging.getLogger(__name__)


def preprocess_function(data):
    result = tokenizer(data['sentence'], truncation=True, max_length=512)
    return result


def compute_metrics(p: EvalPrediction):
    predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    predictions = np.argmax(predictions, axis=1)
    result = metric.compute(predictions=predictions, references=p.label_ids)
    return result


def write_result(row: dict):
    return f'{row["model"]},{row["mean"]} +- {row["std"]}'


if __name__ == '__main__':
    print('\n~~~~~~~~~~ BEGIN ~~~~~~~~~~\n')
    models = ['roberta-base']

    parser = argparse.ArgumentParser()
    parser.add_argument('n_seeds', type=int)
    parser.add_argument('n_train_samples', type=int)
    parser.add_argument('n_validation_samples', type=int)
    parser.add_argument('n_test_samples', type=int)

    my_args = parser.parse_args()

    parser = HfArgumentParser(TrainingArguments)
    # my_args, training_args = parser.parse_args_into_dataclasses()
    # training_args = parser.parse_args_into_dataclasses()
    training_args = TrainingArguments(output_dir='output')
    init_dataset = load_dataset('sst2')

    results = pd.DataFrame(columns=['model_name', 'seed', 'model', 'accuracy'])
    results.index = results['model_name']
    mean_and_std = dict()

    print('\n~~~~~~~~~~ START OF TRAIN ~~~~~~~~~~\n')
    start_training_time = timeit.default_timer()
    for model_name in models:
        metric = load('accuracy')
        training_args.save_strategy = 'no'
        training_args.report_to = ["wandb"]

        for seed in range(my_args.n_seeds):
            set_seed(seed)
            training_args.run_name = f'{model_name}_seed_{seed}'

            config = AutoConfig.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

            dataset = init_dataset.map(preprocess_function, batched=True)

            train_dataset = dataset['train'].select(list(range(my_args.n_train_samples))) \
                if my_args.n_train_samples != -1 \
                else dataset['train']
            validation_dataset = dataset['validation'].select(list(range(my_args.n_validation_samples))) \
                if my_args.n_validation_samples != -1 \
                else dataset['validation']

            print(f'\n~~~~~~~~~~ START TRAIN ON {model_name}, seed = {seed} ~~~~~~~~~~\n')
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
            )

            trainer.train()
            metrics = trainer.evaluate()

            model_data = {'model_name': model_name,
                          'seed': seed,
                          'model': trainer.model,
                          'accuracy': metrics['eval_accuracy']}

            results = pd.concat([results, pd.DataFrame([model_data])], ignore_index=True)
            # finish reporting to wandb
            # wandb.finish()
            # print(f'\n~~~~~~~~~~ FINISH TRAIN ON {model_name}, seed = {seed}, acc = {metrics["eval_accuracy"]} ~~~~~~~~~~\n')

        # model_acc = results.loc['model_name']['accuracy'].to_numpy()

        results.index = results['model_name']
        model_acc = results.loc[model_name]['accuracy']
        mean_and_std[model_name] = model_acc.mean(), model_acc.std()
        # print(f'\n~~~~~~~~~~ FINISH TRAIN ON {model_name}, mean = {model_acc.mean()}, std = {model_acc.std()} ~~~~~~~~~~\n')

    end_training_time = timeit.default_timer()
    # print('\n~~~~~~~~~~ END OF TRAIN ~~~~~~~~~~\n')
    training_time = end_training_time - start_training_time
    results.to_csv('results.csv')

    best_model_name = max(mean_and_std, key=mean_and_std.get)
    accuracy = results.loc[best_model_name]['accuracy']
    best_seed = np.argmax(accuracy.to_numpy()) if my_args.n_seeds > 1 else 0
    results = results.set_index(['seed'], append=True)
    # selected_model = results[best_model_name, best_seed]['model']
    selected_model = results.loc[(best_model_name, best_seed)]['model']

    test_set = dataset['test'].select(list(range(my_args.n_test_samples))) \
        if my_args.n_test_samples != -1 \
        else dataset['test']
    text_classifier = TextClassificationPipeline(model=selected_model.to('cpu'), tokenizer=tokenizer)

    # print('\n~~~~~~~~~~ START OF PREDICT ~~~~~~~~~~\n')
    start_predict_time = timeit.default_timer()
    outputs = text_classifier(test_set['sentence'], batch_size=1)
    end_predict_time = timeit.default_timer()
    # print('\n~~~~~~~~~~ END OF PREDICT ~~~~~~~~~~\n')
    predict_time = end_predict_time - start_predict_time

    with open('res.txt', 'w') as f:
        for key, value in mean_and_std.items():
            f.write(f'{key},{value[0]} +- {value[1]}\n')
        f.write('----\n')
        f.write(f'train time,{training_time}\n')
        f.write(f'predict time,{predict_time}\n')

    with open('predictions.txt', 'w') as f:
        for i in range(len(outputs)):
            f.write(f'{test_set["sentence"][i]}###{outputs[i]["label"][-1]}\n')

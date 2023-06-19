import os
import wandb
import logging
import argparse
from Data.DataPreprocessing import DataPreprocessing
from Model.HumorPredictor import HumorPredictor
from Model.HumorTrainer import HumorTrainer
from datetime import datetime


os.environ["TOKENIZERS_PARALLELISM"] = "false"
wandb.login(key='94ee7285d2d25226f2c969e28645475f9adffbce')
logger = logging.getLogger(__name__)


def my_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('n_train_samples', type=int)
    parser.add_argument('n_test_samples', type=int)
    return parser.parse_args()


if __name__ == '__main__':
    # load datasets
    dpp = DataPreprocessing()
    dataset_names = ['amazon', 'headlines']
    data_path = 'Data/humor_datasets/'
    for name in dataset_names:
        dpp.load_data(data_path + name, name)

    datasets = dpp.get_datasets()

    # train model
    model_params = {
        'model': 'roberta',
        'model_dir': 'roberta-base',
        'train_on_dataset': 'amazon',
        'seed': 18,
    }
    h_trainer = HumorTrainer(model_params, my_parse_args(), datasets)
    trained_model = h_trainer.train()

    # predict labels
    h_predictor = HumorPredictor(trained_model, datasets, h_trainer.get_tokenizer())
    for name in dataset_names:
        h_predictor.predict(name)

    date_str = str(datetime.now().date())
    h_predictor.write_predictions(f'Data/output/predictions/{date_str}')

    # save model
    h_trainer.save_model(f'Data/output/models/{date_str}')

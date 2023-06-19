import os
import wandb
import logging
from Data.DataPreprocessing import DataPreprocessing
from Model.HumorPredictor import HumorPredictor
from Model.HumorTrainer import HumorTrainer
from Utils.utils import print_str, my_parse_args
from datetime import datetime


os.environ["TOKENIZERS_PARALLELISM"] = "false"
wandb.login(key='94ee7285d2d25226f2c969e28645475f9adffbce')
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    # load datasets
    print_str('STARTED RUN')
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
        'seed': 0,
    }
    h_trainer = HumorTrainer(model_params, my_parse_args(), datasets)
    trained_model = h_trainer.train()

    # predict labels
    h_predictor = HumorPredictor(trained_model, h_trainer.get_datasets(), h_trainer.get_tokenizer())
    for name in dataset_names:
        h_predictor.predict(name)

    date_str = str(datetime.now().date())
    h_predictor.write_predictions(f'Data/output/predictions/{date_str}')

    # save model
    h_trainer.save_model(f'Data/output/models/{date_str}')

    print_str('FINISHED RUN')

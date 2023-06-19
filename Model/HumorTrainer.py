import numpy as np
import os
from datetime import datetime
import wandb
from Utils.utils import print_str
from transformers import TrainingArguments, \
    AutoModelForSequenceClassification, set_seed, EvalPrediction, Trainer, \
    AutoTokenizer, AutoConfig

# ===============================      Global Variables:      ===============================
wandb.login(key='94ee7285d2d25226f2c969e28645475f9adffbce')


# ====================================      Class:      ====================================
class HumorTrainer:
    def __init__(self, model_params, args, datasets):
        self._model_params = model_params
        self._init_datasets = datasets

        self._tokenizer = None
        self._metric = None
        self._config = None
        self._model = None
        self._datasets = {}
        self._trainer = None
        self._train_on = self._model_params['train_on_dataset']

        self._hf_args = TrainingArguments(output_dir='output',
                                          save_strategy='no', report_to=['wandb'])

        self._split_sizes = {'train': args.n_train_samples,
                             'test': args.n_test_samples}

    def preprocess_function(self, data):
        result = self._tokenizer(data['sentence'], truncation=True, max_length=512)
        return result

    def compute_metrics(self, p: EvalPrediction):
        predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        predictions = np.argmax(predictions, axis=1)
        result = self._metric.compute(predictions=predictions, references=p.label_ids)
        return result

    def get_split_set(self, dataset_name, split_name):
        num_of_samples = self._split_sizes[split_name]

        split_dataset = self._datasets[dataset_name][split_name].select(list(range(num_of_samples))) \
            if num_of_samples != -1 \
            else self._datasets[dataset_name][split_name]

        return split_dataset

    def get_run_name(self):
        time = datetime.now()
        return '{0}_{1}_{2}:{3}_seed_{4}'.format(self._model_params['model'],
                                                 time.date(), time.hour, time.minute, self._model_params['seed'])

    def get_tokenizer(self):
        return self._tokenizer

    def get_datasets(self):
        return self._datasets

    def process_datasets(self):
        for dataset in self._init_datasets:
            self._datasets[dataset] = self._init_datasets[dataset].map(self.preprocess_function, batched=True)

    def init_model(self):
        set_seed(self._model_params['seed'])
        self._hf_args.run_name = self.get_run_name()
        # TODO which tokenizer, model, config should we use?
        self._config = AutoConfig.from_pretrained(self._model_params['model_dir'])
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_params['model_dir'])
        self._model = AutoModelForSequenceClassification.from_pretrained(self._model_params['model_dir'],
                                                                         config=self._config)
        self.process_datasets()
        # self._datasets[self._train_on] = \
        #     self._init_datasets[self._train_on].map(self.preprocess_function, batched=True)

    def train(self):
        self.init_model()

        self._trainer = Trainer(
            model=self._model,
            args=self._hf_args,
            train_dataset=self.get_split_set(self._train_on, 'train'),
            compute_metrics=self.compute_metrics,
            tokenizer=self._tokenizer
        )

        print_str('STARTED TRAIN ON {0}'.format(self._model_params['model']))
        self._trainer.train()
        print_str('FINISHED TRAIN ON {0}'.format(self._model_params['model']))

        return self._model

    def save_model(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        self._trainer.save_model(os.path + self._model_params['model'])

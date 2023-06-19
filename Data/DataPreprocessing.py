import os
import pandas as pd
from datasets import DatasetDict, Dataset
from sklearn.model_selection import train_test_split


class DataPreprocessing:
    def __init__(self):
        self.datasets = {}

    def load_data(self, path, dataset_name):
        train_path = f'{path}/train.csv'
        test_path = f'{path}/test.csv'

        if os.path.exists(train_path) and os.path.exists(test_path):
            train, test = pd.read_csv(train_path), pd.read_csv(test_path)
        else:
            raise FileExistsError('train or test file does\'nt exists')

        # train = train.sample(frac=1).reset_index(drop=True)
        ds = DatasetDict()
        ds['train'] = Dataset.from_pandas(train)
        ds['test'] = Dataset.from_pandas(test)

        self.datasets[dataset_name] = ds

    def get_datasets(self):
        return self.datasets

    @staticmethod
    def load_amazon():
        path = './original_datasets/amazon/all_data.csv'
        df = pd.read_csv(path)
        df_new = pd.DataFrame()
        df_new['sentence'] = df['question'] + ' [SEP] ' + df['product_description']
        df_new['label'] = df['label']
        df_new = df_new[df_new['sentence'].notna()]
        df_new = df_new.sample(frac=1, random_state=0)
        df_new['idx'] = range(0, len(df_new))
        df_new.to_csv('./humor_datasets/amazon/data.csv', index=False)

        train, test = train_test_split(df_new, test_size=0.2, shuffle=True)
        train.to_csv('./humor_datasets/amazon/train.csv', index=False)
        test.to_csv('./humor_datasets/amazon/test.csv', index=False)

    @staticmethod
    def load_headlines():
        def edit_headline(row):
            headline = row['original']
            edit_word = row['edit']
            res = headline[:headline.index('<')] + edit_word + headline[headline.index('>') + 1:]
            return res

        splits = ['train', 'dev', 'test', 'train_funlines']

        for split in splits:
            path = f'./original_datasets/headlines/{split}.csv'
            df = pd.read_csv(path)
            df = df[df['edit'].notna()]
            df_new = pd.DataFrame()
            df_new['idx'] = df['id']
            df_new['meanGrade'] = df['meanGrade']
            df_new['sentence'] = df.apply(edit_headline, axis=1)
            df_new['label'] = df.apply(lambda row: 1 if row['meanGrade'] >= 1 else 0, axis=1)
            df_new.to_csv(f'./humor_datasets/headlines/{split}.csv', index=False)


if __name__ == '__main__':
    pass
    # DataPreprocessing.load_headlines()
    # DataPreprocessing.load_amazon()

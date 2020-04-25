import inspect
import os
import pandas as pd


class Logger:
    def __init__(self, path):
        self.path = path

    def init_log(self):
        log = pd.DataFrame(columns=['version',
                                    'epoch',
                                    'seed',
                                    'init',
                                    'features',
                                    'datasets',
                                    'train_time',
                                    'timestamp',
                                    'train_loss',
                                    'val_loss',
                                    'train_score',
                                    'val_score',
                                    'train_aprox',
                                    'val_aprox',
                                    'batch_size',
                                    'weight_decay',
                                    'beam',
                                    'description'])
        log.to_csv(self.path, index=False)

    def log(self, model, init_w0, create_feats, load_datasets, description='', manual_cols=None):
        log = pd.read_csv(self.path)
        to_log = [model.version,  # version
                  model.get_log('epoch', epoch='best'),  # epoch
                  model.seed,  # seed
                  inspect.getsource(init_w0),  # init
                  inspect.getsource(create_feats),  # features
                  inspect.getsource(load_datasets),  # datasets
                  model.get_log('train_time', epoch='best'),  # train_time
                  model.get_log('timestamp', epoch='best'),  # timestamp
                  model.get_log('train_loss', epoch='best'),  # train_loss
                  model.get_log('val_loss', epoch='best'),  # val_loss
                  model.get_log('train_score', epoch='best'),  # train_score
                  model.get_log('val_score', epoch='best'),  # val_score
                  model.get_log('train_aprox', epoch='best'),  # train_aprox
                  model.get_log('val_aprox', epoch='best'),  # val_aprox
                  model.get_log('batch_size', epoch='best'),  # batch_size
                  model.get_log('weight_decay', epoch='best'),  # weight_decay
                  model.get_log('beam', epoch='best'),  # beam
                  description]
        ix = 0
        try:
            ix = max(log.index)+1
        except:
            pass
        log.loc[ix] = to_log
        if manual_cols is not None:
            for col, val in manual_cols.items():
                log.loc[ix, col] = val
        log.to_csv(self.path, index=False)

    def remove(self, i):
        log = pd.read_csv(self.path)
        log = log.drop(i)
        log.to_csv(self.path, index=False)

    def leadboard(self, col='val_score', top=5):
        leadboard = pd.read_csv(self.path)
        if col == 'val_score':
            return leadboard.sort_values(col, ascending=False).head(top)
        elif col == 'val_loss':
            return leadboard.sort_values(col, ascending=True).head(top)




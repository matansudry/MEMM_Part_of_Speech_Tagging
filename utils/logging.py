import inspect
import csv
import os


class Logger:
    def __init__(self, path):
        self.path = path

    def init_log(self):
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        with open(os.path.join(self.path, 'log.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['version',
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

    def log(self, model, init_w0, create_feats, load_datasets, description=''):
        with open(os.path.join(self.path, 'log.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([model.version,  # version
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
                             description])  # description

    def leadboard(self, col='val_score'):
        leadboard = pd.read_csv(os.path.join(self.path, 'log.csv'))
        return leadboard.sort_values(col)




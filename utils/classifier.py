import numpy as np
import pandas as pd
import math
import time
from scipy import optimize
import pickle


class Model:
    def __init__(self, w0, tags, feature_vector, seed, score_func, models_path, save):
        self.tags = tags
        self.weights = w0
        self.seed = seed
        self.models_path = models_path
        self.feature_vector = feature_vector
        self.score_func = score_func
        self.log = pd.DataFrame(columns=['epochs',
                                         'train_time',
                                         'timestamp',
                                         'train_loss',
                                         'val_loss',
                                         'train_score',
                                         'val_score',
                                         'batch_size',
                                         'best',
                                        ])
        if save:
            self.save(first=True)

    def __call__(self, t2, t1, w, i):
        # TODO: implement viterbi
        # for tag in self.tags:
            # feat_list_t = feature_vector(t2, t1, w, i, tag, fmt='list')
        return random.choice(self.tags)

    def get_log(self, col='iter', epoch=-1):
        try:
            if epoch == -1:
                index = self.log.tail(1).index[0]
            elif epoch == 'best':
                index = self.log[self.log['best'] == True].tail(1).index[0]
            elif isinstance(epoch, int):
                index = epoch
        except Exception:
            return 0
        if col == 'iter':
            return index
        else:
            try:
                return self.log[col].loc[index]
            except Exception:
                return None

    def save(self, first=False, best=False, epoch=False):
        if first:
            if not os.path.exists(self.models_path):
                os.mkdir(self.models_path)
            if not os.path.exists(os.path.join(self.models_path, naming_scheme(self.version, -1, self.seed, folder=True))):
                os.mkdir(os.path.join(self.models_path, naming_scheme(self.version, -1, self.seed, folder=True)))
        
        with open(os.path.join(self.models_path, naming_scheme(self.version, -1, self.seed)), 'wb') as f:
            pickle.dump(self.weights, f)
        if best:
            with open(os.path.join(self.models_path, naming_scheme(self.version, 'best', self.seed)), 'wb') as f:
                pickle.dump(self.weights, f)
        if epoch:
            with open(os.path.join(self.models_path, naming_scheme(self.version, self.get_log(), self.seed)), 'wb') as f:
                pickle.dump(self.weights, f)

    def predict(self, dataset):
        pred_tags = []
        true_tags = []
        for t2, t1, w, i, t in dataset:
            pred_tags.append(self(t2, t1, w, i))
            true_tags.append(t)
        return pred_tags, true_tags

    def _run(self, dataset, batch_size, weight_decay, iprint, tqdm_bar, train, epochs=None):
        if train:
            self.weights, loss, d_min = optimize.fmin_l_bfgs_b(func=loss_and_grad,
                                                               x0=self.weights,
                                                               args=(dataset,
                                                                     self.feature_vector,
                                                                     weight_decay,
                                                                     False,
                                                                     batch_size,
                                                                     True,
                                                                     self.seed,
                                                                     tqdm_bar),
                                                               maxiter=epochs,
                                                               iprint=iprint)
        else:
            loss = loss_and_grad(v=self.weights,
                                 dataset=dataset,
                                 feature_vector=self.feature_vector,
                                 weight_decay=weight_decay,
                                 loss_only=True,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 seed=self.seed,
                                 tqdm_bar=tqdm_bar)
        
        return loss, self.score(*self.predict(dataset))

    def train(self, epochs, train_dataset, val_dataset=None, batch_size=None, weight_decay=0.0, iprint=-1, save=False, tqdm_bar=False):
        start_epoch = self.get_log()
        start_time = self.get_log('train_time')
        
        tic = time.time()
        epoch = epochs + start_epoch
        
        train_loss, train_score = self._run(train_dataset, batch_size, weight_decay, iprint, tqdm_bar, train=True, epochs=epochs)
        if val_dataset is not None:
            val_loss, val_score = self._run(val_dataset, batch_size, weight_decay, iprint, tqdm_bar, train=False, epochs=None)

        best = val_score > self.get_log('val_score', epoch='best')
        train_time = float(start_time + (time.time() - tic)/60)

        to_log = [epochs,
                  train_time,
                  time.strftime('%H:%M:%S %d-%m-%Y'),
                  train_loss,
                  val_loss,
                  train_score,
                  val_score,
                  batch_size,
                  best,
                  ]
        self.log.loc[epoch] = to_log

        print('epochs {:3d} | train_loss {:.6f} | val_loss {:.6f} | train_score {:.6f} | val_score {:.4f} | train_time {:6.2f} min'
              .format(epochs + start_epoch, train_loss, val_loss, train_score, val_score, train_time))
        # save checkpoint
        if save:
            self.save()
            self.save(epoch=True)
            if best:
                self.save(best=True)


def loss_and_grad(v, dataset, feature_vector, weight_decay=0.0, loss_only=False, batch_size=None, shuffle=False, seed=42, tqdm_bar=False):
    """
    args:
        * v - np.ndarray, model weights
        * dataset - preprocess.Dataset
        * feature_vector - features.FeatureVector
        * weight_decay - lambda regularization
        * tqdm_bar=False - if to display tqdm progress bar
    """
    # regularization term
    loss = -0.5 * weight_decay * np.dot(v, v)
    if not loss_only:
        grad = -weight_decay * v
    
    if batch_size is None:
        batch_size = len(dataset.sentences)
    loader = dataset.load_batch(batch_size, shuffle, seed)
    
    if tqdm_bar:
        from tqdm import tqdm
        loader = tqdm(loader, total=int(batch_size*dataset.words_counter/len(dataset.sentences)))
    
    for t2, t1, w, i, t in loader:
        feat_vec_t, feat_list_t = feature_vector(t2, t1, w, i, t, fmt='both')
        
        # linear term
        loss += sparse_mult(v, feat_list_t)
        # empirical count
        if not loss_only:
            grad += feat_vec_t
        
        # loss normalization term
        sum_exp = 0
        # grad expected_count
        if not loss_only:
            expected_count_nominator_vec = 0.0
            expected_count_denominator = 0.0
        
        for tag in dataset.tags:
            if not loss_only:
                feat_vec_tag, feat_list_tag = feature_vector(t2, t1, w, i, tag, fmt='both')
            else:
                feat_list_tag = feature_vector(t2, t1, w, i, tag, fmt='list')
            
            # grad expected_count
            exp_mult_v_feat = math.exp(sparse_mult(v, feat_list_tag))
            if not loss_only:
                expected_count_nominator_vec += feat_vec_tag*exp_mult_v_feat
                expected_count_denominator += exp_mult_v_feat
            # loss normalization term
            sum_exp += exp_mult_v_feat
        
        loss -= math.log(sum_exp)
        if not loss_only:
            grad -= expected_count_nominator_vec / expected_count_denominator
    
    if loss_only:
        return -loss
    return -loss, -grad


def sparse_mult(np_vec, sparse_list):
    return sum([arg*np_vec[arg] for arg in sparse_list])


def naming_scheme(version, epoch, seed, folder=False):
    if folder:
        return 'V{}'.format(version)
    return os.path.join('V{}'.format(version), 'checkpoint_V{}_SEED{}_E{}.pth'.format(version, epoch, seed))



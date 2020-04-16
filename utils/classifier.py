import numpy as np
import pandas as pd
import math
import time
import random
from scipy import optimize
import pickle


class Model:
    def __init__(self, w0, tags, feature_vector, seed, score_func, models_path, save):
        self.start_fmin_l_bfgs_b_epoch = None
        self.tags = list(tags)
        self.weights = w0
        self.seed = seed
        self.models_path = models_path
        self.feature_vector = feature_vector
        self.score_func = score_func
        self.log = pd.DataFrame(columns=['train_time',
                                         'timestamp',
                                         'train_loss',
                                         'val_loss',
                                         'train_score',
                                         'val_score',
                                         'batch_size',
                                         'best',
                                         'weight_decay',
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

    def train(self, epochs, train_dataset, val_dataset=None, batch_size=None, weight_decay=0.0, iprint=-1, save=False, tqdm_bar=False):
        assert epochs >= 2, 'epochs must be >= 2'
        self.start_fmin_l_bfgs_b_epoch = self.get_log()
        v_min, f_min, d_min = optimize.fmin_l_bfgs_b(func=loss_and_grad,
                                                     x0=self.weights,
                                                     args=(self,
                                                           epochs,
                                                           train_dataset,
                                                           val_dataset,
                                                           True,  # train
                                                           weight_decay,
                                                           batch_size,
                                                           save,
                                                           tqdm_bar),
                                                     maxiter=epochs-1,
                                                     iprint=iprint)
        self.weights = v_min
        return v_min, f_min, d_min


def loss_and_grad(v, model, epochs=None, train_dataset=None, val_dataset=None, train=False, weight_decay=0.0, batch_size=None, save=False, tqdm_bar=False):
    """
    args:
        * v - np.ndarray, model weights
        * model
        * train_dataset=None - preprocess.Dataset
        * val_dataset=None - preprocess.Dataset
        * train=False - if True call in train mode, False for validation mode
            * train mode:
                    * calculates loss and gradient for train_dataset
                    * calculates loss for val_dataset
                    * calculates model.score_func for train_dataset and val_dataset
                    * loggs to model log
                    * returns -loss, -grad
            * validation mode:
                    * calculates -loss for val_dataset
                    * returns -loss
        * weight_decay=0.0 - lambda regularization
        * batch_size=None - batch_size, if None uses full dataset
        * tqdm_bar=False - if to display tqdm progress bar
    """
    if train:
        start_epoch = model.start_fmin_l_bfgs_b_epoch
        epoch = model.get_log() + 1
        start_time = model.get_log('train_time')
        tic = time.time()

        dataset = train_dataset
        # grad regularization term
        grad = -weight_decay * v
    else:
        dataset = val_dataset
    # loss regularization term
    loss = -0.5 * weight_decay * np.dot(v, v)

    batch_size = len(dataset.sentences) if batch_size is None else batch_size
    loader = dataset.load_batch(batch_size, train, model.seed)
    
    if tqdm_bar:
        from tqdm import tqdm
        loader = tqdm(loader, total=int(batch_size*dataset.words_counter/len(dataset.sentences)))
    
    for t2, t1, w, i, t in loader:
        feat_vec_t, feat_list_t = model.feature_vector(t2, t1, w, i, t, fmt='both')
        
        # linear term
        loss += sum(sparse_mult(v, feat_list_t))
        # empirical count
        if train:
            grad += feat_vec_t
        
        # loss normalization term
        sum_exp = 0
        # grad expected_count
        if train:
            expected_count_nominator_vec = 0.0
            expected_count_denominator = 0.0
        
        for tag in dataset.tags:
            if train:
                feat_vec_tag, feat_list_tag = model.feature_vector(t2, t1, w, i, tag, fmt='both')
            else:
                feat_list_tag = model.feature_vector(t2, t1, w, i, tag, fmt='list')
            
            # grad expected_count
            sparse_mult_v_feat = sparse_mult(v, feat_list_tag)
            mult_v_feat = sum(sparse_mult_v_feat)
            try:
                exp_mult_v_feat = math.exp(mult_v_feat)
            except Exception as e:
                print('math error')
                print(f'feat_list_tag={feat_list_tag}')
                print(f'mult_v_feat={mult_v_feat}')
                print(f'sparse_mult_v_feat={sparse_mult_v_feat}')
                raise e
            if train:
                expected_count_nominator_vec += feat_vec_tag*exp_mult_v_feat
                expected_count_denominator += exp_mult_v_feat
            # loss normalization term
            sum_exp += exp_mult_v_feat
        
        loss -= math.log(sum_exp)
        if train:
            grad -= expected_count_nominator_vec / expected_count_denominator
    
    if train:
        train_loss = -loss/batch_size
        val_loss = loss_and_grad(v, model, val_dataset=val_dataset, train=False, weight_decay=weight_decay, tqdm_bar=tqdm_bar)/len(val_dataset.sentences)
        
        train_score = model.score_func(*model.predict(train_dataset))
        val_score = model.score_func(*model.predict(val_dataset))
        
        best = val_score > model.get_log('val_score', epoch='best')
        train_time = float(start_time + (time.time() - tic)/60)

        to_log = [train_time,
                  time.strftime('%H:%M:%S %d-%m-%Y'),
                  train_loss,
                  val_loss,
                  train_score,
                  val_score,
                  batch_size,
                  best,
                  weight_decay,
                  ]
        model.log.loc[epoch] = to_log

        # epoch progress prints
        print('epoch {:3d}/{:d} | train_loss {:.6f} | val_loss {:.6f} | train_score {:.6f} | val_score {:.4f} | train_time {:6.2f} min'
              .format(epoch, epochs + start_epoch, train_loss, val_loss, train_score, val_score, train_time))
        # save checkpoint
        if save:
            model.save()
            model.save(epoch=True)
            if best:
                model.save(best=True)
        
        return -loss, -grad
    return -loss


def sparse_mult(np_vec, sparse_list):
    return [arg*np_vec[arg] for arg in sparse_list]


def naming_scheme(version, epoch, seed, folder=False):
    if folder:
        return 'V{}'.format(version)
    return os.path.join('V{}'.format(version), 'checkpoint_V{}_SEED{}_E{}.pth'.format(version, epoch, seed))


def viterbi():
    
    pass







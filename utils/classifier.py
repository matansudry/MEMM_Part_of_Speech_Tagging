import os
import math
import time
import random
import numpy as np
import pandas as pd
from scipy import optimize


def load_model(version, models_path, epoch=-1, seed=42, prints=True):
    import dill

    model_path = os.path.join(models_path, naming_scheme(version, epoch, seed))
    try:
        with open(model_path, "rb") as f:
            model = dill.load(f)
    except Exception:
        print("Loading Error")
        raise
    
    if prints and model.get_log() > 0:
        print("model version:", model.version)
        print("epochs: {}\ntrain_time: {:.3f}\n".format(model.get_log('epoch'), model.get_log('train_time')))
        print("last train_loss: {:.6f}".format(model.get_log('train_loss')))
        print("last val_loss: {:.6f}".format(model.get_log('val_loss')))
        print("last train_score: {:.6f}".format(model.get_log('train_score')))
        print("last val_score: {:.6f}".format(model.get_log('val_score')))
        print("best val_score: {:.4f} at epoch {:d}".format(model.get_log('val_score'), model.get_log('epoch', epoch='best')))
    return model


def accuracy(pred_tags, true_tags):
    correct = 0
    total = 0
    for preds, tags in zip(pred_tags, true_tags):
        for pred, tag in zip(preds, tags):
            total += 1
            if pred == tag:
                correct += 1
    return float(correct)/total


def rnd(model, sentence, beam):
    return [random.choice(model.tags) for i in sentence]

    
class Model:
    def __init__(self, version, w0, tags, inference, feature_vector, seed, score_func, models_path, save):
        self.version = version
        self.start_fmin_l_bfgs_b_epoch = None
        self.tags = list(tags)
        self.weights = w0
        self.inference = inference
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
                                         'beam',
                                        ])
        if save:
            self.save(first=True)

    def __call__(self, sentence, beam):
        return self.inference(self, sentence, beam)

    def get_log(self, col='epoch', epoch=-1):
        try:
            if epoch == -1:
                index = self.log.tail(1).index[0]
            elif epoch == 'best':
                index = self.log[self.log['best'] == True].tail(1).index[0]
            elif isinstance(epoch, int):
                index = epoch
        except Exception:
            return 0
        if col == 'epoch':
            return index
        else:
            try:
                return self.log[col].loc[index]
            except Exception:
                return None

    def save(self, first=False, best=False, epoch=False):
        import dill
        if first:
            if not os.path.exists(self.models_path):
                os.mkdir(self.models_path)
            if not os.path.exists(os.path.join(self.models_path, naming_scheme(self.version, -1, self.seed, folder=True))):
                os.mkdir(os.path.join(self.models_path, naming_scheme(self.version, -1, self.seed, folder=True)))
        
        with open(os.path.join(self.models_path, naming_scheme(self.version, -1, self.seed)), 'wb') as f:
            dill.dump(self, f)
        if best:
            with open(os.path.join(self.models_path, naming_scheme(self.version, 'best', self.seed)), 'wb') as f:
                dill.dump(self, f)
        if epoch:
            with open(os.path.join(self.models_path, naming_scheme(self.version, self.get_log(), self.seed)), 'wb') as f:
                dill.dump(self, f)

    def predict(self, sentences, beam, tqdm_bar=False):
        if tqdm_bar:
            from tqdm import tqdm
            sentences = tqdm(sentences)
        pred_tags = []
        true_tags = []
        for sentence in sentences:
            pred_tags.append(self(sentence[0], beam))
            true_tags.append(sentence[1])
        return pred_tags, true_tags

    def train(self, epochs, train_dataset, val_dataset=None, batch_size=None, weight_decay=0.0, iprint=-1, save=False, tqdm_bar=False, beam=None, train_aprox=None, val_aprox=None):
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
                                                           tqdm_bar,
                                                           beam,
                                                           train_aprox,
                                                           val_aprox),
                                                     maxiter=epochs-1,
                                                     iprint=iprint)
        self.weights = v_min
        return v_min, f_min, d_min


def loss_and_grad(v, model, epochs=None, train_dataset=None, val_dataset=None, train=False, weight_decay=0.0,
                  batch_size=None, save=False, tqdm_bar=False, beam=None, train_aprox=None, val_aprox=None):
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
        model.weights = v
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

        if train_aprox is None:
            train_aprox = len(train_dataset.sentences)

        if val_aprox is None:
            val_aprox = len(val_dataset.sentences)

        train_score = model.score_func(*model.predict(train_dataset.sentences[:train_aprox], beam))
        val_score = model.score_func(*model.predict(val_dataset.sentences[:val_aprox], beam))
        
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
                  beam,
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
    return [np_vec[arg] for arg in sparse_list]


def naming_scheme(version, epoch, seed, folder=False):
    if folder:
        return 'V{}'.format(version)
    return os.path.join('V{}'.format(version), 'checkpoint_V{}_E{:03d}_SEED{}.pth'.format(version, epoch, seed))


def viterbi(model, sentence, beam):
    pi_bp = {}  # pi_bp[i][(t1, t)]
    pi_bp[-1] = {('*', '*'): ('*', 1)}
    t1_list, t_list = ['*'], ['*']
    for i, word in enumerate(sentence):
        t2_list, t1_list, t_list = t1_list, t_list, model.tags
        pi_bp[i] = {}
        for t in t_list:
            argmax = {}
            for t2, t1 in pi_bp[i-1]:
                argmax[t] = softmax(model, t2, t1, sentence, i, t) * pi_bp[i-1][(t2, t1)][1]
            pi_bp[i][(t1, t)] = sorted(list(argmax.items()), key=lambda x: x[1], reverse=True)[0]
        if beam:
            pi_bp[i] = dict(sorted(list(pi_bp[i].items()), key=lambda x: x[1][1], reverse=True)[:beam])

    tags = []
    for i, _ in enumerate(sentence):
        k = len(sentence) - i - 1
        tags.insert(0, sorted(list(pi_bp[k].items()), key=lambda x: x[1][1], reverse=True)[0][1][0])

    return tags


def softmax(model, t2, t1, w, i, t):
    q_denominator = 0.0
    for tag in model.tags:
        feat_list_tag = model.feature_vector(t2, t1, w, i, tag, fmt='list')
        
        sparse_mult_v_feat = sparse_mult(model.weights, feat_list_tag)
        mult_v_feat = sum(sparse_mult_v_feat)
        try:
            exp_mult_v_feat = math.exp(mult_v_feat)
        except Exception as e:
            print('math error')
            print(f'feat_list_tag={feat_list_tag}')
            print(f'mult_v_feat={mult_v_feat}')
            print(f'sparse_mult_v_feat={sparse_mult_v_feat}')
            raise e
        if tag == t:
            q_nominator = exp_mult_v_feat
        q_denominator += exp_mult_v_feat
    return q_nominator/q_denominator



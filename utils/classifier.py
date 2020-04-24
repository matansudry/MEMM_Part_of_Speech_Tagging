import os
import math
import time
import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt
from . import metrics 


def naming_scheme(version, epoch, seed, folder=False):
    if folder:
        return 'V{}'.format(version)
    if isinstance(epoch, str):
        return os.path.join('V{}'.format(version), 'checkpoint_V{}_E{}_SEED{}.pth'.format(version, epoch, seed))
    return os.path.join('V{}'.format(version), 'checkpoint_V{}_E{:03d}_SEED{}.pth'.format(version, epoch, seed))


def viterbi(model, sentence, beam=1):
    """
    args:
        * model
        * sentence - words only, no tags
        * beam - beam size
    """
    bp_pi = {}  # bp_pi[i][(t1, t)] = (t2, p)
    bp_pi[-1] = {('*', '*'): (None, 1)}
    for i in range(len(sentence)):
        bp_pi[i] = {}
        for t in model.tags:
            argmax_t1 = {}
            for (t2, t1) in bp_pi[i-1]:
                if t1 not in argmax_t1:
                    argmax_t1[t1] = {}
                argmax_t1[t1][t2] = _softmax(model, t2, t1, sentence, i, t) * bp_pi[i-1][(t2, t1)][1]
            for t1 in argmax_t1:
                bp_pi[i][(t1, t)] = sorted(list(argmax_t1[t1].items()), key=lambda x: x[1], reverse=True)[0]
        if beam:
            bp_pi[i] = dict(sorted(list(bp_pi[i].items()), key=lambda x: x[1][1], reverse=True)[:beam])

    tags = []
    argmax_k = None
    for i in range(len(sentence)-2):
        argmax_k = sorted(list(bp_pi[i+2].items()), key=lambda x: x[1][1], reverse=True)[0]
        tags.append(argmax_k[1][0])
    argmax_k = sorted(list(bp_pi[max(list(bp_pi.keys()))].items()), key=lambda x: x[1][1], reverse=True)[0]
    tags.extend([argmax_k[0][0], argmax_k[0][1]])

    return tags[-len(sentence):]

class Model:
    def __init__(self, version, w0, tags, feature_vector, inference=viterbi, seed=42, score_func=metrics.accuracy,
                 models_path='models', max_weights_history=5, save=False):
        self.version = version
        self.start_fmin_l_bfgs_b_epoch = None
        self.tags = list(tags)
        self.weights = w0
        self.weights_history = []
        self.max_weights_history = max_weights_history
        self.inference = inference
        self.seed = seed
        self.models_path = models_path
        self.feature_vector = feature_vector
        self.val_predictions = None
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
                                         'train_aprox',
                                         'val_aprox',
                                         'beam',
                                        ])
        if save:
            self.save()

    def __call__(self, sentence, beam):
        return self.inference(self, sentence, beam)

    def avg_weights(self, save=True):
        self.weights = np.zeros(len(self.weights))
        for weights in self.weights_history:
            self.weights += weights
        self.weights /= len(self.weights_history)
        if save:
            self.save(avg=True)

    def load(self, weights=True, weights_history=True, feature_vector=True, log=True, epoch=-1, prints=True):
        loaded_model = load_model(version=self.version, models_path=self.models_path, epoch=epoch, seed=self.seed, prints=prints)
        if weights:
            self.weights = loaded_model.weights
        if weights_history:
            self.weights_history = loaded_model.weights_history
        if feature_vector:
            self.feature_vector = loaded_model.feature_vector
        if log:
            self.log = loaded_model.log
    
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

    def plot(self, attributes, plot_title, y_label, scale='linear', basey=10):
        if not self.get_log():
            print("model have not trained yet")
            return
        epochs = self.log.index
        to_plot = []
        for attribute in attributes:
            to_plot.append(self.log[attribute])
        min_e = np.min(epochs)
        max_e = np.max(epochs)
        for data in to_plot:
            plt.plot(epochs, data)
        plt.xlim(min_e - (max_e - min_e)*0.02, max_e + (max_e - min_e)*0.02)
        plt.xlabel('epoch')
        plt.ylabel(y_label)
        if scale == 'log':
            plt.yscale(scale, basey=basey)
        else:
            plt.yscale(scale)
        plt.legend(attributes)
        plt.title(plot_title)
        plt.savefig('{}.png'.format(plot_title), dpi=200)
        plt.show()

    def save(self, best=False, epoch=False, avg=False, manual_path=None):
        try:
            import dill
        except:
            print('model save failed, could not import dill')
            return

        if manual_path is not None:
            with open(manual_path, 'wb') as f:
                dill.dump(self, f)
            return

        if not os.path.exists(self.models_path):
            os.mkdir(self.models_path)
        if not os.path.exists(os.path.join(self.models_path, naming_scheme(self.version, -1, self.seed, folder=True))):
            os.mkdir(os.path.join(self.models_path, naming_scheme(self.version, -1, self.seed, folder=True)))
        
        with open(os.path.join(self.models_path, naming_scheme(self.version, -1, self.seed)), 'wb') as f:
            dill.dump(self, f)
        if best:
            with open(os.path.join(self.models_path, naming_scheme(self.version, 'best', self.seed)), 'wb') as f:
                dill.dump(self, f)
        if avg:
            with open(os.path.join(self.models_path, naming_scheme(self.version, 'avg', self.seed)), 'wb') as f:
                dill.dump(self, f)
        if epoch:
            with open(os.path.join(self.models_path, naming_scheme(self.version, self.get_log(), self.seed)), 'wb') as f:
                dill.dump(self, f)

    def predict(self, sentences, beam, tqdm_bar=False):
        if tqdm_bar:
            try:
                from tqdm import tqdm
                sentences = tqdm(sentences)
            except:
                pass
        pred_tags = []
        true_tags = []
        for sentence in sentences:
            pred_tags.append(self(sentence[0], beam))
            true_tags.append(sentence[1])
        return pred_tags, true_tags

    def train(self, epochs, train_dataset, val_dataset=None, batch_size=None, weight_decay=0.0, save=False, tqdm_bar=False, beam=None, train_aprox=None, val_aprox=None, batch_growth=None):
        """
        args:
            * epochs - train epochs
            * train_dataset
            * val_dataset=None
            * batch_size=None
            * weight_decay=0.0 - lamda regularization parameter
            * save=False - save model after each epoch
            * tqdm_bar=False - display tqdm progress bars
            * beam=None - viterbi beam size for model evaluation during training
            * train_aprox=None - max train samples to aproximate train_score
            * val_aprox=None - max val samples to aproximate val_score
            * batch_growth=None - num of epochs to double batch_size
        """
        assert epochs >= 2, 'epochs must be >= 2'
        self.start_fmin_l_bfgs_b_epoch = self.get_log()
        np.random.seed(self.seed)
        v_min, f_min, d_min = optimize.fmin_l_bfgs_b(func=_loss_and_grad,
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
                                                           val_aprox,
                                                           batch_growth),
                                                     maxfun=epochs-1,
                                                     maxiter=epochs-1,
                                                     iprint=-1)
        self.weights_history.append(v_min)
        self.weights_history = self.weights_history[-self.max_weights_history:]
        self.weights = v_min
        return v_min, f_min, d_min


def _loss_and_grad(v, model, epochs, train_dataset, val_dataset, train, weight_decay,
                  batch_size, save, tqdm_bar, beam, train_aprox, val_aprox, batch_growth):
    """
    see Model.train documentation
    """
    if train:
        model.weights = v
        model.weights_history.append(v)
        model.weights_history = model.weights_history[-model.max_weights_history:]
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

    if batch_size is None or batch_size == 0 or batch_size > len(dataset.sentences):
        batch_size = len(dataset.sentences)
    
    if batch_growth is not None and batch_growth > 0:
        batch_size = min(batch_size*(2**((epoch - start_epoch - 1)//batch_growth)), len(dataset.sentences))
    
    loader = dataset.load_batch(batch_size, train, model.seed)

    if tqdm_bar:
        try:
            from tqdm import tqdm
            loader = tqdm(loader, total=int(batch_size*dataset.words_counter/len(dataset.sentences)))
        except:
            pass

    for t2, t1, w, i, t in loader:
        feat_vec_t, feat_list_t = model.feature_vector(t2, t1, w, i, t, fmt='both')

        # linear term
        loss += sum(_sparse_mult(v, feat_list_t))
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
            _sparse_mult_v_feat = _sparse_mult(v, feat_list_tag)
            mult_v_feat = sum(_sparse_mult_v_feat)
            exp_mult_v_feat = math.exp(mult_v_feat)
            if train:
                expected_count_nominator_vec += feat_vec_tag*exp_mult_v_feat
                expected_count_denominator += exp_mult_v_feat
            # loss normalization term
            sum_exp += exp_mult_v_feat

        loss -= math.log(sum_exp)
        if train:
            grad -= expected_count_nominator_vec / expected_count_denominator

    loss /= batch_size
    if train:
        grad /= batch_size

    if train:
        train_loss = -loss
        val_loss = _loss_and_grad(v, model, 0, None, val_dataset, False, weight_decay, None, False, tqdm_bar, 0, 0, 0, None) if val_dataset is not None else 0.0

        train_aprox = len(train_dataset.sentences) if train_aprox is None else train_aprox
        val_aprox = len(val_dataset.sentences) if (val_aprox is None and val_dataset is not None) else val_aprox

        train_score = model.score_func(*model.predict(train_dataset.sentences[:train_aprox], beam))
        val_score = model.score_func(*model.predict(val_dataset.sentences[:val_aprox], beam)) if val_dataset is not None else 0.0

        if val_aprox > 0:
            best = val_score > model.get_log('val_score', epoch='best')
        else:
            best = val_loss < model.get_log('val_loss', epoch='best')

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
                  train_aprox,
                  val_aprox,
                  beam,
                  ]  # train_aprox=None, val_aprox=None
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


def _sparse_mult(np_vec, sparse_list):
    return [np_vec[arg] for arg in sparse_list]


def _softmax(model, t2, t1, w, i, t):
    """
    helper func for viterbi
    """
    q_denominator = 0.0
    for tag in model.tags:
        feat_list_tag = model.feature_vector(t2, t1, w, i, tag, fmt='list')

        _sparse_mult_v_feat = _sparse_mult(model.weights, feat_list_tag)
        mult_v_feat = sum(_sparse_mult_v_feat)
        exp_mult_v_feat = math.exp(mult_v_feat)
        if tag == t:
            q_nominator = exp_mult_v_feat
        q_denominator += exp_mult_v_feat
    return q_nominator/q_denominator


def load_model(version='', models_path='', epoch=-1, seed=42, prints=True, from_file=None):
    try:
        import dill
    except:
        print('model load failed, could not import dill')
        return

    if from_file is None:
        model_path = os.path.join(models_path, naming_scheme(version, epoch, seed))
    else:
        model_path = from_file
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



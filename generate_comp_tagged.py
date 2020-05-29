import os
import time
import pickle
import numpy as np
from utils import preprocess, features, classifier, metrics, logging


seed = 42
np.random.seed(seed)
models_path = 'models'
data_path = 'data'
model1_file_name = 'final_model1.pkl'
model2_file_name = 'final_model2.pkl'
comp1_pred_file_name = f'comp_m1_321128258.wtag'
comp2_pred_file_name = f'comp_m2_321128258.wtag'


def load_datasets():
    train1_dataset = preprocess.Dataset(os.path.join(data_path, 'train1.wtag'))
    test1_dataset = preprocess.Dataset(os.path.join(data_path, 'test1.wtag'))
    comp1_dataset = preprocess.Dataset(os.path.join(data_path, 'comp1.words'), labeled=False, tags=train1_dataset.tags)

    train2_dataset = preprocess.Dataset(os.path.join(data_path, 'train2.wtag'))
    comp2_dataset = preprocess.Dataset(os.path.join(data_path, 'comp2.words'), labeled=False, tags=train2_dataset.tags)
    
    return train1_dataset, test1_dataset, comp1_dataset, train2_dataset, comp2_dataset

def load_feature_vectors(train1_dataset, train2_dataset):
    group_thresholds = {
        # -------------------------------- feature --------------------- | -- Threshold --
        lambda t2, t1, w, i, t: tuple([w[i].lower(), t]):                         0,     # mandatory feature f100
        lambda t2, t1, w, i, t: tuple([w[i][-4:].lower(), t]):                    5,     # mandatory feature f101
        lambda t2, t1, w, i, t: tuple([w[i][-3:].lower(), t]):                    5,     # mandatory feature f101
        lambda t2, t1, w, i, t: tuple([w[i][-2:].lower(), t]):                    5,     # mandatory feature f101
        lambda t2, t1, w, i, t: tuple([w[i][-1:].lower(), t]):                    5,     # mandatory feature f101
        lambda t2, t1, w, i, t: tuple([w[i][:4].lower(), t]):                     5,     # mandatory feature f102
        lambda t2, t1, w, i, t: tuple([w[i][:3].lower(), t]):                     5,     # mandatory feature f102
        lambda t2, t1, w, i, t: tuple([w[i][:2].lower(), t]):                     5,     # mandatory feature f102
        lambda t2, t1, w, i, t: tuple([w[i][:1].lower(), t]):                     5,     # mandatory feature f102
        lambda t2, t1, w, i, t: tuple([t2, t1, t]):                               1,     # mandatory feature f103
        lambda t2, t1, w, i, t: tuple([t1, t]):                                   1,     # mandatory feature f104
        lambda t2, t1, w, i, t: tuple([t]):                                       1,     # mandatory feature f105
        lambda t2, t1, w, i, t: tuple([w[i].islower(), t]):                       1,     # mandatory feature has_uppercase
        lambda t2, t1, w, i, t: tuple([any(char.isdigit() for char in w[i]), t]): 1,     # mandatory feature has_digits
        lambda t2, t1, w, i, t: tuple([w[i-1].lower(), t]):                       20,
        lambda t2, t1, w, i, t: tuple([w[i+1].lower(), t]):                       20,
        lambda t2, t1, w, i, t: tuple([w[i+1][:3].lower(), t]):                   20,
        lambda t2, t1, w, i, t: tuple([w[i-1][:3].lower(), t]):                   20,
        lambda t2, t1, w, i, t: tuple([w[i+1][:2].lower(), t]):                   20,
        lambda t2, t1, w, i, t: tuple([w[i-1][:2].lower(), t]):                   20,
        lambda t2, t1, w, i, t: tuple([w[i+1][-3:].lower(), t]):                  20,
        lambda t2, t1, w, i, t: tuple([w[i-1][-3:].lower(), t]):                  20,
        lambda t2, t1, w, i, t: tuple([w[i+1][-2:].lower(), t]):                  20,
        lambda t2, t1, w, i, t: tuple([w[i-1][-2:].lower(), t]):                  20,
        lambda t2, t1, w, i, t: tuple([w[i].isalnum(), t]):                       10,
        lambda t2, t1, w, i, t: tuple([w[i].isalpha(), t]):                       10,
        lambda t2, t1, w, i, t: tuple([w[i].isascii(), t]):                       10,
        lambda t2, t1, w, i, t: tuple([w[i].isdecimal(), t]):                     10,
        lambda t2, t1, w, i, t: tuple([w[i].isdigit(), t]):                       10,
        lambda t2, t1, w, i, t: tuple([w[i].isnumeric(), t]):                     10,
        lambda t2, t1, w, i, t: tuple([w[i].istitle(), t]):                       10,
        lambda t2, t1, w, i, t: tuple([w[i].isupper(), t]):                       10,
        lambda t2, t1, w, i, t: tuple([len(w[i]), t]):                            10,
    }

    feature_vector1 = features.create_feature_vector(dataset=train1_dataset,
                                                     group_thresholds=group_thresholds,
                                                     pruning=True,
                                                     get_stats=False,
                                                     assertions=False,
                                                     calls_counter=False)

    for feat in feature_vector1.feats:
        print('feat_group:', feat, '| feats:', len(feat))
    print('feat_groups:', len(feature_vector1.feats), '| total_feats:', len(feature_vector1))
    print()
    
    feature_vector2 = features.create_feature_vector(dataset=train2_dataset,
                                                     group_thresholds=group_thresholds,
                                                     pruning=True,
                                                     get_stats=False,
                                                     assertions=False,
                                                     calls_counter=False)

    for feat in feature_vector2.feats:
        print('feat_group:', feat, '| feats:', len(feat))
    print('feat_groups:', len(feature_vector2.feats), '| total_feats:', len(feature_vector2))
    
    return feature_vector1, feature_vector2

def retrain_model1(train1_dataset, feature_vector1):
    np.random.seed(seed)
    model1 = classifier.Model(version=1,
                              w0=np.random.rand(len(feature_vector1)),
                              tags=train1_dataset.tags,
                              inference=classifier.viterbi,
                              feature_vector=feature_vector1,
                              score_func=metrics.accuracy,
                              models_path=models_path,
                              max_weights_history=0,
                              save=False)

    _, _, _ = model1.train(epochs=92,  # training epochs
                           train_dataset=train1_dataset,
                           val_dataset=None,
                           batch_size=256,  # initial batch_size for loader
                           weight_decay=0.0,  # lamda regularization parameter
                           save=False,  # save model during training (requires dill module)
                           tqdm_bar=False,  # display tqdm progress bars
                           beam=1,  # viterbi beam size for model evaluation during training
                           train_aprox=0,  # aproximate train_score with first train_aprox train samples
                           val_aprox=0,  # aproximate val_score with first val_aprox train samples 
                           batch_growth=4)  # double batch_size every batch_growth epochs
    
    model1.feature_vector = None
    with open(model1_file_name, 'wb') as f:
        pickle.dump(model1, f)
    
    model1.feature_vector = feature_vector1
    
    return model1

def retrain_model2(train2_dataset, feature_vector2):
    np.random.seed(seed)
    version = 2
    train_save = False  # save model after each training epoch, if False model will need to be saved manually
    beam = 1  # viterbi beam size for model evaluation during training
    train_aprox = 0  # aproximate train_score with first train_aprox train samples
    val_aprox = 0  # aproximate val_score with first val_aprox train samples 
    weight_decay = 0.0  # lamda regularization parameter
    init_batch_size = 250  # batch_size for loader
    batch_growth = 0
    epochs = 43  # training epochs
    tqdm_bar = False  # display tqdm progress bars

    model2 = classifier.Model(version=2,
                              w0=np.random.rand(len(feature_vector2)),
                              tags=train2_dataset.tags,
                              inference=classifier.viterbi,
                              feature_vector=feature_vector2,
                              score_func=metrics.accuracy,
                              models_path=models_path,
                              max_weights_history=0,
                              save=False)

    _, _, _ = model2.train(epochs=43,  # training epochs
                           train_dataset=train2_dataset,
                           val_dataset=None,
                           batch_size=250,  # initial batch_size for loader
                           weight_decay=0.0,  # lambda regularization parameter
                           save=False,  # save model during training (requires dill module)
                           tqdm_bar=False,  # display tqdm progress bars
                           beam=1,  # viterbi beam size for model evaluation during training
                           train_aprox=0,  # aproximate train_score with first train_aprox train samples
                           val_aprox=0,  # aproximate val_score with first val_aprox train samples
                           batch_growth=0)      # double batch_size every batch_growth epochs
    
    model2.feature_vector = None
    with open(model2_file_name, 'wb') as f:
        pickle.dump(model2, f)
    
    model2.feature_vector = feature_vector2
    
    return model2

def load_trained_models(train1_dataset, train2_dataset):
    with open(model1_file_name, "rb") as f:
        model1 = pickle.load(f)

    with open(model2_file_name, "rb") as f:
        model2 = pickle.load(f)

    model1.feature_vector, model2.feature_vector = load_feature_vectors(train1_dataset, train2_dataset)
        
    return model1, model2

def save_wtag(dataset, comp_pred_tags, version):
    comp_wtag_list = []
    for i in range(len(dataset.sentences)):
        joined_sentence = []
        assert len(dataset.sentences[i][0]) == len(comp_pred_tags[i]), \
            f'i={i}, len(dataset.sentences[i][0])={len(dataset.sentences[i][0])}, len(comp_pred_tags[i])={len(comp_pred_tags[i])}'
        for word, pred in zip(dataset.sentences[i][0], comp_pred_tags[i]):
            joined_sentence.append('_'.join([word, pred]))
        comp_wtag_list.append(' '.join(joined_sentence))

    with open(f'comp_m{version}_321128258.wtag', 'w') as f:
        for row in comp_wtag_list:
            f.write(row)
            f.write('\n')


def main():
    np.random.seed(seed)
    train1_dataset, test1_dataset, comp1_dataset, train2_dataset, comp2_dataset = load_datasets()
    
    # # retrain models
    # feature_vector1, feature_vector2 = load_feature_vectors(train1_dataset, train2_dataset)
    # model1 = retrain_model1(train1_dataset, feature_vector1) 
    # model2 = retrain_model2(train2_dataset, feature_vector2)

    # load models
    model1, model2 = load_trained_models(train1_dataset, train2_dataset)

    test1_pred_tags, test1_true_tags = model1.predict(test1_dataset.sentences, beam=1, tqdm_bar=False)
    test1_accuracy = model1.score_func(test1_pred_tags, test1_true_tags)
    test1_confusion_matrix, test1_tags_accuracy = metrics.confusion_matrix(train1_dataset.tags, test1_pred_tags, test1_true_tags)

    worst10_test1_confusion_matrix = test1_confusion_matrix.loc[list(test1_tags_accuracy.keys())[:10], list(test1_tags_accuracy.keys())[:10]]
    worst10_test1_tags_accuracy = list(test1_tags_accuracy.items())[:10]
    print('test1_accuracy:', test1_accuracy)
    print('worst10_test1_tags_accuracy:', worst10_test1_tags_accuracy)
    print('worst10_test1_confusion_matrix:\n', worst10_test1_confusion_matrix)

    comp1_pred_tags = model1.predict(comp1_dataset.sentences, beam=5, tqdm_bar=False)[0]
    save_wtag(comp1_dataset, comp1_pred_tags, 1)

    comp2_pred_tags = model2.predict(comp2_dataset.sentences, beam=1, tqdm_bar=False)[0]
    save_wtag(comp2_dataset, comp2_pred_tags, 2)

    # insert path to comp1 and comp2 true tagged files (in a .wtag format) to perform accuracy evaluation
    comp1_tagged_path = 'comp1_tagged.wtag'
    comp2_tagged_path = 'comp2_tagged.wtag'

    try:
        comp1_true_dataset = preprocess.Dataset(comp1_tagged_path)
        comp1_accuracy = model1.score_func(comp1_pred_tags, [sentence[1] for sentence in comp1_true_dataset.sentences])
        print(f'comp1_accuracy={comp1_accuracy}')
    except:
        pass

    try:
        comp2_true_dataset = preprocess.Dataset(comp2_tagged_path)
        comp2_accuracy = model2.score_func(comp2_pred_tags, [sentence[1] for sentence in comp2_true_dataset.sentences])
        print(f'comp2_accuracy={comp2_accuracy}')
    except:
        pass


if __name__ == "__main__":
    main()




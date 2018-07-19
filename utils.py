################################################################################################################
# Author: Suthee Chaidaroon
# schaidaroon@scu.edu
################################################################################################################

import numpy as np
import os
import scipy.io
from dotmap import DotMap
from tqdm import tqdm
from rank_metrics import *

################################################################################################################
class MedianHashing(object):
    
    def __init__(self):
        self.threshold = None
        self.latent_dim = None
    
    def fit(self, X):
        self.threshold = np.median(X, axis=0)
        self.latent_dim = X.shape[1]
        
    def transform(self, X):
        assert(X.shape[1] == self.latent_dim)
        binary_code = np.zeros(X.shape)
        for i in range(self.latent_dim):
            binary_code[np.nonzero(X[:,i] < self.threshold[i]),i] = 0
            binary_code[np.nonzero(X[:,i] >= self.threshold[i]),i] = 1
        return binary_code.astype(int)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

################################################################################################################
def Load_Dataset(filename):
    dataset = scipy.io.loadmat(filename)
    x_train = dataset['train']
    x_test = dataset['test']
    x_cv = dataset['cv']
    y_train = dataset['gnd_train']
    y_test = dataset['gnd_test']
    y_cv = dataset['gnd_cv']
    
    data = DotMap()
    data.n_trains = y_train.shape[0]
    data.n_tests = y_test.shape[0]
    data.n_cv = y_cv.shape[0]
    data.n_tags = y_train.shape[1]
    data.n_feas = x_train.shape[1]

    ## Convert sparse to dense matricesimport numpy as np
    train = x_train
    nz_indices = np.where(np.sum(train, axis=1) > 0)[0]
    train = train[nz_indices, :]
    train_len = np.sum(train > 0, axis=1)
    train_len = np.squeeze(np.asarray(train_len))

    test = x_test
    test_len = np.sum(test > 0, axis=1)
    test_len = np.squeeze(np.asarray(test_len))

    if x_cv is not None:
        cv = x_cv
        cv_len = np.sum(cv > 0, axis=1)
        cv_len = np.squeeze(np.asarray(cv_len))
    else:
        cv = None
        cv_len = None
        
    gnd_train = y_train[nz_indices, :]
    gnd_test = y_test
    gnd_cv = y_cv

    data.train = train
    data.test = test
    data.cv = cv
    data.train_len = train_len
    data.test_len = test_len
    data.cv_len = cv_len
    data.gnd_train = gnd_train
    data.gnd_test = gnd_test
    data.gnd_cv = gnd_cv
    
    return data

################################################################################################################

class TopDoc(object):
    def __init__(self, data_fn, is_train=False):
        self.data_fn = data_fn
        self.is_train = is_train
        self.db = self.load(data_fn, is_train)
        
    def load(self, fn, is_train):
        db = {}
        with open(fn) as in_data:
            for line in in_data:
                line = line.strip()
                first, rest = line.split(':')

                topk = list(map(int, rest.split(',')))
                
                docId = int(first)
                if is_train:
                    db[docId] = topk[1:]
                else:
                    db[docId] = topk
        return db
    
    def getTopK(self, docId, topK):
        return self.db[docId][:topK]

    def getTopK_Noisy(self, docId, topK, topCandidates):
        candidates = self.db[docId][:topCandidates]
        candidates = np.random.permutation(candidates)
        return candidates[:topK]

################################################################################################################

def run_topK_retrieval_experiment_GPU_batch_train(codeTrain, codeTest, 
                                                  gnd_train, gnd_test, batchSize=500, TopK=100):
    
    import torch
    #from tqdm import tqdm_notebook as tqdm
    assert (codeTrain.shape[1] == codeTest.shape[1])
    assert (gnd_train.shape[1] == gnd_test.shape[1])
    assert (codeTrain.shape[0] == gnd_train.shape[0])
    assert (codeTest.shape[0] == gnd_test.shape[0])
    
    n_bits = codeTrain.shape[1]
    n_train = codeTrain.shape[0]
    n_test = codeTest.shape[0]

    topScores = torch.cuda.ByteTensor(n_test, TopK + batchSize).fill_(n_bits+1)
    topIndices = torch.cuda.LongTensor(n_test, TopK + batchSize).zero_()

    testBinmat = torch.cuda.ByteTensor(codeTest).unsqueeze_(2)
    for batchIdx in tqdm(range(0, n_train, batchSize), ncols=0):
        s_idx = batchIdx
        e_idx = min(batchIdx + batchSize, n_train)
        numCandidates = e_idx - s_idx

        batch_codeTrain = codeTrain[s_idx:e_idx].T
        trainBinmat = torch.cuda.ByteTensor(batch_codeTrain).unsqueeze_(0)
        trainBinmat = trainBinmat.expand(testBinmat.size(0), n_bits, trainBinmat.size(2))

        testBinmatExpand = testBinmat.expand_as(trainBinmat)

        scores = (trainBinmat ^ testBinmatExpand).sum(dim=1) #.type(torch.cuda.FloatTensor)
        indices = torch.from_numpy(np.arange(s_idx, e_idx)).cuda().unsqueeze_(0).expand(n_test, numCandidates)

        topScores[:, -numCandidates:] = scores
        topIndices[:, -numCandidates:] = indices

        topScores, newIndices = topScores.sort(dim=1)
        topIndices = torch.gather(topIndices, 1, newIndices)

    # Compute Precision
    Indices = topIndices[:,:TopK]

    y_test = np.argmax(gnd_test, axis=1)
    testLabels = torch.cuda.ByteTensor(y_test).unsqueeze_(1)
    testLabels = testLabels.expand(n_test, TopK)

    y_train = np.argmax(gnd_train, axis=1)
    trainLabels = torch.cuda.ByteTensor(y_train) #.unsqueeze_(1)
    topTrainLabels = [torch.index_select(trainLabels, 0, Indices[idx]).unsqueeze_(0) for idx in range(0, n_test)]
    topTrainLabels = torch.cat(topTrainLabels, dim=0)

    relevances = (testLabels == topTrainLabels).type(torch.cuda.ShortTensor)
    true_positive = relevances.sum(dim=1).type(torch.cuda.FloatTensor)
    true_positive = true_positive.div_(100)
    prec_at_k = torch.mean(true_positive)
    
    tqdm.write('Prec@K = {:.4f}'.format(prec_at_k))
    return prec_at_k, None
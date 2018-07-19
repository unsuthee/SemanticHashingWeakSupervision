################################################################################################################
# Author: Suthee Chaidaroon
# schaidaroon@scu.edu
################################################################################################################

from dotmap import DotMap
import numpy as np
import scipy.io
import pickle
import os
from utils import *
from tqdm import *
import sklearn.preprocessing
from scipy import sparse

import argparse

##################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpunum", help="GPU number to train the model.")
parser.add_argument("-d", "--dataset", help="Name of the dataset.")
parser.add_argument("-b", "--nbits", help="Number of bits of the embedded vector.", type=int)
parser.add_argument("--train_batch_size", default=100, type=int)
parser.add_argument("--test_batch_size", default=100, type=int)
parser.add_argument("--transform_batch_size", default=100, type=int)
parser.add_argument("--num_epochs", default=30, type=int)
parser.add_argument("--lr", default=0.001, type=float)

args = parser.parse_args()

if not args.gpunum:
    parser.error("Need to provide the GPU number.")
    
if not args.dataset:
    parser.error("Need to provide the dataset.")

if not args.nbits:
    parser.error("Need to provide the dataset.")
        
##################################################################################################

DATASET = args.dataset
data = Load_Dataset("data/{}.mat".format(DATASET))

##################################################################################################

label_binarizer = sklearn.preprocessing.LabelBinarizer()
label_binarizer.fit(range(data.n_tags))

gnd_train = data.gnd_train
gnd_test = data.gnd_test

##################################################################################################

print(gnd_train.shape)
print(gnd_test.shape)
print('num train:{}'.format(data.n_trains))
print('num test:{}'.format(data.n_tests))

train_topk_docs_db = TopDoc('bm25/{}_train_top101.txt'.format(DATASET), is_train=True)
test_topk_docs_db = TopDoc('bm25/{}_test_top101.txt'.format(DATASET), is_train=False)

##################################################################################################

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter

class NbrRegDoc(nn.Module):
    
    def __init__(self, vocabSize, latentDim, dropoutProb=0.):
        super(NbrRegDoc, self).__init__()
        
        self.hidden_dim = 1000
        self.vocabSize = vocabSize
        self.latentDim = latentDim
        
        self.dtype = torch.cuda.FloatTensor

        self.fc1 = nn.Linear(self.vocabSize, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc31 = nn.Linear(self.hidden_dim, self.latentDim)
        self.fc32 = nn.Linear(self.hidden_dim, self.latentDim)
        self.dropout = nn.Dropout(p=dropoutProb)
        
        self.nn_fc1 = nn.Linear(self.vocabSize, self.hidden_dim)
        self.nn_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.fc41 = nn.Linear(self.latentDim, self.vocabSize)
        self.nn_fc41 = nn.Linear(self.latentDim, self.vocabSize)

    def encode_nn(self, batch_nn_mat):
        n_batches = batch_nn_mat.shape[0]
        batch_nn_docs = Variable(torch.from_numpy(batch_nn_mat).type(self.dtype))
        batch_nn_docs = batch_nn_docs.view(-1, self.vocabSize)
        
        h1 = self.relu(self.nn_fc1(batch_nn_docs))
        h2 = self.relu(self.nn_fc2(h1))
        h3 = h2.view(n_batches, -1, self.hidden_dim)
        
        h_nn = torch.mean(h3, dim=1)
        return h_nn
        
    def encode(self, document_mat, batch_nn_mat):
        documents = Variable(torch.from_numpy(document_mat).type(self.dtype))
        h1 = self.relu(self.fc1(documents))
        h2 = self.relu(self.fc2(h1))
        h3 = self.dropout(h2)
        
        nn_h = self.encode_nn(batch_nn_mat)
        
        z_mu = self.fc31(h3 + nn_h)
        z_logvar = self.sigmoid(self.fc32(h3 + nn_h))
        return z_mu, z_logvar
    
    def decode(self, Z):
        word_prob = self.fc41(Z)
        word_prob = self.log_softmax(word_prob)
        
        nn_word_prob = self.nn_fc41(Z)
        nn_word_prob = self.log_softmax(nn_word_prob)
        
        return word_prob, nn_word_prob
    
    def reparametrize(self, mu, logvar):
        std = torch.sqrt(torch.exp(logvar))
        
        if self.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def forward(self, document_mat, batch_nn_mat):
        mu, logvar = self.encode(document_mat, batch_nn_mat)
        z = self.reparametrize(mu, logvar)
        prob_w, nn_word_prob = self.decode(z)
        return prob_w, nn_word_prob, mu, logvar
    
def calculate_KL_loss(mu, logvar):
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element, dim=1)
    KLD = torch.mean(KLD).mul_(-0.5)
    return KLD

def compute_reconstr_loss(log_word_prob, document_mat):
    loss = None
    
    for idx, doc_vec in enumerate(document_mat):
        word_indices = doc_vec.nonzero()
        word_indices = Variable(torch.from_numpy(word_indices[0]).type(torch.cuda.LongTensor))
        pred_logprob = torch.gather(log_word_prob[idx], 0, word_indices)
        
        if loss is None:
            loss = -torch.sum(pred_logprob) 
        else:
            loss.add_(-torch.sum(pred_logprob))

    return loss / document_mat.shape[0]

def batch_compute_NN_reconstr_loss(log_word_prob, batch_nn_docs):
    batch_nn_docs = np.sum(batch_nn_docs, axis=1)
    nn_loss = None
    
    for docIdx, nn_docs in enumerate(batch_nn_docs):
        word_indices = np.nonzero(nn_docs)
        word_indices = Variable(torch.cuda.LongTensor(word_indices[0]))
        pred_logprob = torch.gather(log_word_prob[docIdx], 0, word_indices)

        if nn_loss is None:
            nn_loss = -torch.sum(pred_logprob) 
        else:
            nn_loss.add_(-torch.sum(pred_logprob))
    
    return nn_loss / float(len(batch_nn_docs))

##################################################################################################

GPU_NUM = args.gpunum
NUM_BITS = args.nbits
TEST_BATCH_SIZE = args.test_batch_size

os.environ["CUDA_VISIBLE_DEVICES"]=GPU_NUM

model = NbrRegDoc(data.n_feas, NUM_BITS, dropoutProb=0.1)
model.cuda()

nn_TOP_K = 20
nn_TOP_Candidates = nn_TOP_K

##################################################################################################

def transform(docs, nn_top_k, nn_Corpus, nn_DocRetrieval, batch_size=500):
    n_docs = len(docs)
    Z = None
    model.eval()
    for batch_idx in range(0, n_docs, batch_size):
        s_idx = batch_idx
        e_idx = min(s_idx + batch_size, n_docs)

        batch_docs = docs[s_idx:e_idx]
        
        # get all neighbor documents 
        batch_nn_docs = []
        for docId in range(s_idx, e_idx):
            nn_docList = nn_DocRetrieval.getTopK(docId, nn_top_k)
            nn_docs = [nn_Corpus[nn_docId] for nn_docId in nn_docList]
            batch_nn_docs.append(nn_docs)
        batch_nn_docs = np.stack(batch_nn_docs)
  
        mu, _ = model.encode(batch_docs, batch_nn_docs)
        if Z is None:
            Z = mu.cpu().data.numpy()
        else:
            Z = np.concatenate((Z, mu.data.cpu().numpy()), axis=0)
    return Z

##################################################################################################

transform_batch_size = args.transform_batch_size
test_batch_size = args.test_batch_size
TopK = 100

def run_test():
    model.eval()
    test_loss = 0

    batch_size = transform_batch_size
    train = data.train.toarray()
    test = data.test.toarray()
    
    tqdm.write('\rTransform training set ...', end='')
    z_train = transform(train, nn_TOP_K, train, train_topk_docs_db, batch_size=batch_size)
    tqdm.write('\rTransform testing set ...', end='')
    z_test = transform(test, nn_TOP_K, train, test_topk_docs_db, batch_size=batch_size)
    
    tqdm.write('\rRun Experiment ...', end='')
    medHash = MedianHashing()
    cbTrain = medHash.fit_transform(z_train)
    cbTest = medHash.transform(z_test)
    
    gnd_train = data.gnd_train.toarray()
    gnd_test = data.gnd_test.toarray()
    
    return run_topK_retrieval_experiment_GPU_batch_train(cbTrain, cbTest, 
                                      gnd_train, gnd_test,
                                      batchSize=test_batch_size, TopK=100)

##################################################################################################

optimizer = optim.Adam(model.parameters(), lr=args.lr)

BestPrec = 0.
BestRound = 0

BATCH_SIZE = args.train_batch_size
NUM_EPOCHS = args.num_epochs

use_noisy = False

# KL weight annealing
klWeight = 0.
klStepSize = 1 / 5000.
    
for iteration in range(1, NUM_EPOCHS + 1):
    model.train()
    train_loss = []

    pbar = tqdm(total=data.n_trains, ncols=0)
    for batch_idx in range(0, data.n_trains, BATCH_SIZE):
        s_idx = batch_idx
        e_idx = min(s_idx + BATCH_SIZE, data.n_trains)
        
        batch_train = data.train[s_idx:e_idx]
        batch_train = batch_train.toarray()
        
        batch_nn_docs = []
        for docId in range(s_idx, e_idx):
            if not use_noisy:
                nn_docList = train_topk_docs_db.getTopK(docId, nn_TOP_K)
            else:
                nn_docList = train_topk_docs_db.getTopK_Noisy(docId, nn_TOP_K, nn_TOP_Candidates)
            nn_docs = data.train[nn_docList].toarray()
            batch_nn_docs.append(nn_docs)
        batch_nn_docs = np.stack(batch_nn_docs)

        optimizer.zero_grad()
        
        log_word_prob, nn_log_word_prob, mu, logvar = model(batch_train, batch_nn_docs)
        
        reconstr_loss = compute_reconstr_loss(log_word_prob, batch_train)
        nn_reconstr_loss = batch_compute_NN_reconstr_loss(nn_log_word_prob, batch_nn_docs)

        kl_loss = calculate_KL_loss(mu, logvar)
        loss = reconstr_loss + nn_reconstr_loss + (klWeight * kl_loss)
        
        loss.backward()
        optimizer.step()

        klWeight = min(klWeight + klStepSize, 1.)            
        train_loss.append(loss.item())

        pbar.set_description("{}: NbrReg+Doc Best Round:{} Prec:{:.4f} AvgLoss:{:.3f}"
                             .format(iteration, BestRound, BestPrec, np.mean(train_loss)))
        pbar.update(len(batch_train))
        

    pbar.close()
    
    prec, _ = run_test()
    BestPrec = max(BestPrec, prec)
    
    if BestPrec == prec:
        BestRound = iteration

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
from tqdm import tqdm
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

##################################################################################################

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter

class VDSH(nn.Module):
    
    def __init__(self, vocabSize, latentDim, dropoutProb=0.):
        super(VDSH, self).__init__()
        
        self.hidden_dim = 1000
        self.vocabSize = vocabSize
        self.latentDim = latentDim
        
        self.dtype = torch.cuda.FloatTensor

        self.fc1 = nn.Linear(self.vocabSize, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc31 = nn.Linear(self.hidden_dim, self.latentDim)
        self.fc32 = nn.Linear(self.hidden_dim, self.latentDim)
        self.dropout = nn.Dropout(p=dropoutProb)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.fc41 = nn.Linear(self.latentDim, self.vocabSize)
        
    def encode(self, document_mat):
        documents = Variable(torch.from_numpy(document_mat).type(self.dtype))
        h1 = self.relu(self.fc1(documents))
        h2 = self.relu(self.fc2(h1))
        h3 = self.dropout(h2)
        
        z_mu = self.fc31(h3)
        z_logvar = self.sigmoid(self.fc32(h3))
        return z_mu, z_logvar
    
    def decode(self, Z):
        word_prob = self.fc41(Z)
        return self.log_softmax(word_prob)
        
    def reparametrize(self, mu, logvar):
        std = torch.sqrt(torch.exp(logvar))
        
        if self.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def forward(self, document_mat):
        mu, logvar = self.encode(document_mat)
        z = self.reparametrize(mu, logvar)
        prob_w = self.decode(z)
        return prob_w, mu, logvar
    
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

##################################################################################################

GPU_NUM = args.gpunum
NUM_BITS = args.nbits
TEST_BATCH_SIZE = args.test_batch_size

os.environ["CUDA_VISIBLE_DEVICES"]=GPU_NUM

model = VDSH(data.n_feas, NUM_BITS, dropoutProb=0.1)
model.cuda()

def transform(doc_mat, batch_size=500):
    Z = None
    model.eval()
    for idx in range(0, doc_mat.shape[0], batch_size):
        if idx + batch_size < doc_mat.shape[0]:
            batch_train = doc_mat[idx:idx+batch_size]
        else:
            batch_train = doc_mat[idx:]
            
        mu, _ = model.encode(batch_train)
        if Z is None:
            Z = mu.cpu().data.numpy()
        else:
            Z = np.concatenate((Z, mu.cpu().data.numpy()), axis=0)
    return Z

TopK = 100
def run_test():
    model.eval()
    test_loss = 0

    batch_size = args.transform_batch_size
    z_train = transform(data.train.toarray())
    z_test = transform(data.test.toarray())
    
    medHash = MedianHashing()
    cbTrain = medHash.fit_transform(z_train)
    cbTest = medHash.transform(z_test)
    
    gnd_train = data.gnd_train.toarray()
    gnd_test = data.gnd_test.toarray()
    
    return run_topK_retrieval_experiment_GPU_batch_train(cbTrain, cbTest, 
                                  gnd_train, gnd_test,
                                  batchSize=TEST_BATCH_SIZE, TopK=100)

##################################################################################################

optimizer = optim.Adam(model.parameters(), lr=args.lr)

BATCH_SIZE = args.train_batch_size
NUM_EPOCHS = args.num_epochs

# KL weight annealing
klWeight = 0.
klStepSize = 1 / 5000.

BestPrec = 0.
BestRound = 0

for iteration in range(1, NUM_EPOCHS + 1):
    model.train()
    train_loss = []

    pbar = tqdm(total=data.n_trains, ncols=0)
    for idx in range(0, data.n_trains, BATCH_SIZE):
        if idx + BATCH_SIZE < data.n_trains:
            batch_train = data.train[idx:idx+BATCH_SIZE]
        else:
            batch_train = data.train[idx:]

        batch_train = batch_train.toarray()
        
        optimizer.zero_grad()
        
        word_prob, mu, logvar = model(batch_train)
        
        reconstr_loss = compute_reconstr_loss(word_prob, batch_train)
        kl_loss = calculate_KL_loss(mu, logvar)
        loss = reconstr_loss + (klWeight * kl_loss)
        
        loss.backward()
        optimizer.step()

        klWeight = min(klWeight + klStepSize, 1.)            
        train_loss.append(loss.item())

        pbar.set_description("{}: VDSH Best Round:{} Prec:{:.4f} AvgLoss:{:.3f}"
                             .format(iteration, BestRound, BestPrec, np.mean(train_loss)))
        pbar.update(len(batch_train))

    pbar.close()
    
    prec, _ = run_test()
    BestPrec = max(BestPrec, prec)
    
    if BestPrec == prec:
        BestRound = iteration
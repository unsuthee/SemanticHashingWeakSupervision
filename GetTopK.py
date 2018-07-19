################################################################################################################
# Author: Suthee Chaidaroon
# schaidaroon@scu.edu
################################################################################################################
import numpy as np
import os
from utils import *
from tqdm import tqdm
import scipy.io
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

#################################################################################################################
def GetTopK_UsingCosineSim(outfn, queries, documents, TopK, queryBatchSize=10, docBatchSize=100):
    
    n_docs = documents.shape[0]
    n_queries = queries.shape[0]
    query_row = 0
    
    with open(outfn, 'w') as out_fn:
        for q_idx in tqdm(range(0, n_queries, queryBatchSize), desc='Query', ncols=0):
            query_batch_s_idx = q_idx
            query_batch_e_idx = min(query_batch_s_idx + queryBatchSize, n_queries)

            queryMats = torch.cuda.FloatTensor(queries[query_batch_s_idx:query_batch_e_idx].toarray())
            queryNorm2 = torch.norm(queryMats, 2, dim=1)
            queryNorm2.unsqueeze_(1)
            queryMats.unsqueeze_(2)

            scoreList = []
            indicesList = []

            #print('{}: perform cosine sim ...'.format(q_idx))
            for idx in tqdm(range(0, n_docs, docBatchSize), desc='Doc', leave=False, ncols=0):
                batch_s_idx = idx
                batch_e_idx = min(batch_s_idx + docBatchSize, n_docs)
                n_doc_in_batch = batch_e_idx - batch_s_idx

                #if batch_s_idx > 1000:
                #    break

                candidateMats = torch.cuda.FloatTensor(documents[batch_s_idx:batch_e_idx].toarray())

                candidateNorm2 = torch.norm(candidateMats, 2, dim=1)
                candidateNorm2.unsqueeze_(0)

                candidateMats.unsqueeze_(2)
                candidateMats = candidateMats.permute(2, 1, 0)

                # compute cosine similarity
                queryMatsExpand = queryMats.expand(queryMats.size(0), queryMats.size(1), candidateMats.size(2))
                candidateMats = candidateMats.expand_as(queryMatsExpand)

                cos_sim_scores = torch.sum(queryMatsExpand * candidateMats, dim=1) / (queryNorm2 * candidateNorm2)

                K = min(TopK, n_doc_in_batch)
                scores, indices = torch.topk(cos_sim_scores, K, dim=1, largest=True)

                del cos_sim_scores
                del queryMatsExpand
                del candidateMats
                del candidateNorm2

                scoreList.append(scores)
                indicesList.append(indices + batch_s_idx)

            all_scores = torch.cat(scoreList, dim=1)
            all_indices = torch.cat(indicesList, dim=1)
            _, indices = torch.topk(all_scores, TopK, dim=1, largest=True)

            topK_indices = torch.gather(all_indices, 1, indices)
            #all_topK_indices.append(topK_indices)
            #all_topK_scores.append(scores)

            del queryMats
            del queryNorm2
            del scoreList
            del indicesList

            topK_indices = topK_indices.cpu().numpy()
            for row in topK_indices:
                out_fn.write("{}:".format(query_row))
                outtext = ','.join([str(col) for col in row])
                out_fn.write(outtext)
                out_fn.write('\n')
                query_row += 1

            torch.cuda.empty_cache()

#################################################################################################################

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpunum")
parser.add_argument("--dataset")
parser.add_argument("--usetrain", action='store_true')

args = parser.parse_args()
if args.gpunum:
    print("Use GPU #:{}".format(args.gpunum))
    gpunum = args.gpunum
else:
    print("Use GPU #0 as a default gpu")
    gpunum = "0"
    
os.environ["CUDA_VISIBLE_DEVICES"]=gpunum

if args.dataset:
    print("load {} dataset".format(args.dataset))
    dataset = args.dataset
else:
    parser.error("Need to provide the dataset.")
    
data = Load_Dataset("data/ng20.mat")

print("num train:{} num tests:{}".format(data.n_trains, data.n_tests))

if args.usetrain:
    print("use train as a query corpus")
    query_corpus = data.train
    out_fn = "bm25/{}_train_top101.txt".format(dataset)
else:
    print("use test as a query corpus")
    query_corpus = data.test
    out_fn = "bm25/{}_test_top101.txt".format(dataset)

print("save the result to {}".format(out_fn))
GetTopK_UsingCosineSim(out_fn, query_corpus, data.train, TopK=101, queryBatchSize=500, docBatchSize=100)
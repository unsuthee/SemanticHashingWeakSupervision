# Deep Semantic Text Hashing with Weak Supervision (SIGIR'18)
## Author: Suthee Chaidaroon
This is a pyTorch implementation of two models described in [Deep Semantic Text Hashing with Weak Supervision](https://dl.acm.org/citation.cfm?id=3210090).

## Requirements
Python 3.6 and PyTorch 0.4. 

## Datasets
We use 4 datasets in this paper: 20Newsgroups, DBPedia, YahooAnswers, and AG's news. You can download the original datasets from the link provided in the paper. For your convenience, the preprocessed dataset can be downloaded from [here](https://drive.google.com/open?id=1uYITf8XPuULPLm0nQg3A7nxDzF79U2-C). These datasets are bag-of-words using BM25 weighting. The k-nearest neighbors for each document in both train and test collections are also provided.

It is important to create two data folders to train the models. The first one is "data" directory where it stores all bag-of-words datasets. The second folder is "bm25" where we use to save all the k-nearest neighbors data.

## Run the program
We provided 3 models in this repo: VDSH[1], NbrReg, and NbrReg+Doc. To train the model, use the following commands:

To train NbrReg model:

```
python train_NbrReg.py -g 0 -b 32 -d ng20 --epoch 30 --batch_size 100
```

To train NbrReg+Doc model:

```
python train_NbrRegDoc.py -g 0 -b 32 -d ng20 --epoch 30 --batch_size 100
```

To train VDSH model:

```
python train_VDSH.py -g 0 -b 32 -d ng20 --epoch 30 --batch_size 100
```

## Custom datasets
If you are interested in training our models on your custom datasets, you need to ensure that the dataset is in a bag-of-words format. You also need to generate a k-nearest neighbors file by running:

To create kNN for a train set:
```
python topK.py -d your_custom_dataset -g 0 --use_train
```

To create kNN for a test set:
```
python topK.py -d your_custom_dataset -g 0 --use_train
```

## References
[1] Chaidaroon, Suthee, and Yi Fang. "Variational deep semantic hashing for text documents." Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval. ACM, 2017.



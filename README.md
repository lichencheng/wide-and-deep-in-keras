# wide-and-deep-in-keras
1. Abstract：This project is just a practical test for predicting the CTR in a recommendation system. Google first proposed the wide&deep      model in the paper "Wide & Deep Learning for Recommender Systems" https://arxiv.org/pdf/1606.07792.pdf. In China, many Internet            companies declared that they have made some improvements in wide&deep for serving their own business. Unfortunately， I connot get the      codes, so I searched the Github and found some codes in keras. I like implementing the deep learning model in keras, whcih is really      easy to read. But I found some mistakes (maybe I misunderstood) in the codes, so I learnt from the codes and read the original paper      wide&deep. Finally, I tried to write a simple model for wide&deep.

2. Introduction: There have been many study materials for wide&deep on the Internet, so I omit the illustrations of the details of the        paper. However, during the implementation of my project I found there are two critical points, which must be paid attention to. One is    the data preprocess (oneHot for category features and MinMaxScale for continuous features), the other is the joint training. I must        cite the original paper,
   " Note that there is a distinction between
   joint training and ensemble. In an ensemble, individual
   models are trained separately without knowing each
   other, and their predictions are combined only at inference
   time but not at training time. In contrast, joint training
   optimizes all parameters simultaneously by taking both the
   wide and deep part as well as the weights of their sum into
   account at training time. There are implications on model
   size too: For an ensemble, since the training is disjoint, each
   individual model size usually needs to be larger (e.g., with
   more features and transformations) to achieve reasonable
   accuracy for an ensemble to work. In comparison, for joint
   training the wide part only needs to complement the weaknesses
   of the deep part with a small number of cross-product
   feature transformations, rather than a full-size wide model.". 
   It is easy to implement the wide&deep in a ensemble model (I indeed did). Do not feed the wide part with the full-size featue instead      of the cross-product feature transformations. In my experiment, the loss got higher and the accuracy got lower when the full-size          featue was used to feed to the wide part.

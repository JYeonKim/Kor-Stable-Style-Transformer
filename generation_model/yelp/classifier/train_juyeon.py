import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
import os
import random

from transformers import *
# gpt_tokenizer = GPT2Tokenizer.from_pretrained('skt/kogpt2-base-v2')
# gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt_tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')

from tqdm import tqdm
import json

# import pdb; pdb.set_trace()

## 초기화
from dis_model import *
dismodel = findattribute().cuda()
# dismodel = findattribute()
dismodel.train()

import torch.optim as optim

# 네이버 리뷰 데이터
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
# from konlpy.tag import Okt
from tqdm import tqdm

# from tensorboardX import SummaryWriter
# summary = SummaryWriter(logdir='./logs')

def main():
    # f = open('../gpt_yelp_vocab.json')
    # token2num = json.load(f)

    # num2token = {}
    # for key, value in token2num.items():
    #     num2token[value] = key
    # f.close()

    # load data file
    urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
    urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")
    train_data = pd.read_table('ratings_train.txt')
    test_data = pd.read_table('ratings_test.txt')

    # 중복 제거 및 공백 제거
    train_data.drop_duplicates(subset=['document'], inplace=True)
    train_data = train_data.dropna(how = 'any')
    train_data = train_data.iloc[:,1:] # id column 제거

    train_neg = train_data[train_data['label'] == 0]['document']
    train_neg_dataset = np.array(train_neg)
    neg_len = len(train_neg_dataset)

    train_pos = train_data[train_data['label'] == 1]['document']
    train_pos_dataset = np.array(train_pos)
    pos_len = len(train_pos_dataset)

    ##################################################

    # data_path = "/DATA/joosung/sentiment_data/Sentiment-and-Style-Transfer-master/data" # customize data path
    # yelp_neg_path = data_path + "/yelp/sentiment.train.0"
    # yelp_neg_open = open(yelp_neg_path, "r")
    # yelp_neg_dataset = yelp_neg_open.readlines()
    # neg_len = len(yelp_neg_dataset)
    # yelp_neg_open.close()

    # yelp_pos_path = data_path + "/yelp/sentiment.train.1"
    # yelp_pos_open = open(yelp_pos_path, "r")
    # yelp_pos_dataset = yelp_pos_open.readlines()
    # pos_len = len(yelp_pos_dataset)
    # yelp_pos_open.close()

    """training parameter"""
    cls_initial_lr = 0.001
    cls_trainer = optim.Adamax(dismodel.cls_params, lr=cls_initial_lr) # initial 0.001
    max_grad_norm = 25
    batch = 1
    # epoch = 5
    epoch = 1
    stop_point = pos_len*epoch

    pre_epoch = 0
    for start in tqdm(range(0, stop_point)):
        ## learing rate decay
        now_epoch = (start+1)//pos_len
        if now_epoch == 4:
            cls_initial_lr = cls_initial_lr/2
            cls_trainer = optim.Adamax(dismodel.cls_params, lr=cls_initial_lr) # initial 0.001

        """data start point"""
        neg_start = start%neg_len
        pos_start = start%pos_len

        """data setting"""
        # neg_sentence = yelp_neg_dataset[neg_start].strip()
        # pos_sentence = yelp_pos_dataset[pos_start].strip()
        neg_sentence = train_neg_dataset[neg_start].strip()
        pos_sentence = train_pos_dataset[pos_start].strip()

        neg_labels = [] # negative labels
        neg_labels.append([1,0])
        neg_attribute = torch.from_numpy(np.asarray(neg_labels)).type(torch.FloatTensor).cuda()
        # neg_attribute = torch.from_numpy(np.asarray(neg_labels)).type(torch.FloatTensor)

        pos_labels = [] # positive labels
        pos_labels.append([0,1])
        pos_attribute = torch.from_numpy(np.asarray(pos_labels)).type(torch.FloatTensor).cuda()
        # pos_attribute = torch.from_numpy(np.asarray(pos_labels)).type(torch.FloatTensor)

        sentences = [neg_sentence, pos_sentence]
        attributes = [neg_attribute, pos_attribute]

        """data input"""
        for i in range(2):
            # k=0: negative, k=1: positive
            sentence = sentences[i]
            attribute = attributes[i] # for generate

            token_idx = torch.tensor(gpt_tokenizer.encode(sentence)).unsqueeze(0).cuda()
            # token_idx = torch.tensor(gpt_tokenizer.encode(sentence)).unsqueeze(0)
            dis_out = dismodel.discriminator(token_idx)
            # try:
            #     dis_out = dismodel.discriminator(token_idx)
            # except:
            #     import pdb; pdb.set_trace()

            """calculation loss & traning"""
            # training using discriminator loss
            cls_loss = dismodel.cls_loss(attribute, dis_out)
            # summary.add_scalar('discriminator loss', cls_loss.item(), start)

            cls_trainer.zero_grad()
            cls_loss.backward() # retain_graph=True
            grad_norm = torch.nn.utils.clip_grad_norm_(dismodel.cls_params, max_grad_norm)
            cls_trainer.step()

        """savining point"""
        if (start+1)%pos_len == 0:
            # random.shuffle(yelp_neg_dataset)
            # random.shuffle(yelp_pos_dataset)
            random.shuffle(train_neg_dataset)
            random.shuffle(train_pos_dataset)
            save_model((start+1)//pos_len)
    save_model('final') # final_model

def save_model(iter):
    if not os.path.exists('models/'):
        os.makedirs('models/')
    torch.save(dismodel.state_dict(), 'models/cls_model_{}'.format(iter))


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()


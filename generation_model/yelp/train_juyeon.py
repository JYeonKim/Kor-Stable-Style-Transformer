import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
import os
import random

from transformers import *
# gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt_tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')
from tqdm import tqdm
import json

# 추가
import pandas as pd
import numpy as np

## 초기화
from gen_model import *
genmodel = styletransfer().cuda()
genmodel.train()

sys.path.insert(0, "/content/drive/MyDrive/Colab Notebooks/Stable-Style-Transformer/generation_model/yelp/classifier")
from dis_model import *
dismodel = findattribute().cuda()
# dismodel_name='cls_model_3'
dismodel_name='cls_model_final'
dismodel.load_state_dict(torch.load('models/{}'.format(dismodel_name)))
dismodel.eval()


import torch.optim as optim

def main():
    # f = open('gpt_yelp_vocab.json')
    # token2num = json.load(f)

    # num2token = {}
    # for key, value in token2num.items():
    #     num2token[value] = key
    # f.close()

    # data_path = "/DATA/joosung/sentiment_data/Sentiment-and-Style-Transfer-master/data"
    # train_yelp_neg_path = data_path + "/yelp/sentiment.train.0"
    # train_yelp_neg_open = open(train_yelp_neg_path, "r")
    # train_yelp_neg_dataset = train_yelp_neg_open.readlines()
    # yelp_neg_dataset = train_yelp_neg_dataset

    # neg_len = len(yelp_neg_dataset)
    # train_yelp_neg_open.close()

    # train_yelp_pos_path = data_path + "/yelp/sentiment.train.1"
    # train_yelp_pos_open = open(train_yelp_pos_path, "r")
    # train_yelp_pos_dataset = train_yelp_pos_open.readlines()
    # yelp_pos_dataset = train_yelp_pos_dataset

    # pos_len = len(yelp_pos_dataset)
    # train_yelp_pos_open.close()

    ##################################

    # load data file
    train_data = pd.read_table('ratings_train.txt')

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

    ##################################

    """training parameter"""
    aed_initial_lr = 0.00001
    gen_initial_lr = 0.001
    aed_trainer = optim.Adamax(genmodel.aed_params, lr=aed_initial_lr) # initial 0.0005
    gen_trainer = optim.Adamax(genmodel.aed_params, lr=gen_initial_lr) # initial 0.0001
    max_grad_norm = 20
    batch = 1
    # epoch = 6
    epoch = 1
    stop_point = pos_len*epoch

    pre_epoch = 0
    for start in tqdm(range(0, stop_point)):
        ## learing rate decay
        now_epoch = (start+1)//pos_len

        """data start point"""
        neg_start = start%neg_len
        pos_start = start%pos_len

        """data setting"""
        neg_sentence = train_neg_dataset[neg_start].strip()
        pos_sentence = train_pos_dataset[pos_start].strip()

        neg_labels = [] # negative labels
        neg_labels.append([1,0])
        neg_attribute = torch.from_numpy(np.asarray(neg_labels)).type(torch.FloatTensor).cuda()

        pos_labels = [] # positive labels
        pos_labels.append([0,1])
        pos_attribute = torch.from_numpy(np.asarray(pos_labels)).type(torch.FloatTensor).cuda()

        sentences = [neg_sentence, pos_sentence]
        attributes = [neg_attribute, pos_attribute]
        sentiments = [0, 1]
        with torch.autograd.set_detect_anomaly(True):
            """data input"""
            for i in range(2):
                # k=0: negative, k=1: positive
                sentence = sentences[i]
                attribute = attributes[i] # for decoder
                fake_attribute = attributes[abs(1-i)] # for generate
    #             sentiment = sentiments[i] # for delete
                token_idx = torch.tensor(gpt_tokenizer.encode(sentence)).unsqueeze(0).cuda()

                # delete model
                max_len = int(token_idx.shape[1]/2)
                dis_out = dismodel.discriminator(token_idx)
                sentiment = dis_out.argmax(1).cpu().item() ## 변경점 for delete

                del_idx = token_idx
                for k in range(max_len):
                    del_idx = dismodel.att_prob(del_idx, sentiment)
                    dis_out = dismodel.discriminator(del_idx)
                    sent_porb = F.softmax(dis_out, 1).squeeze(0)[sentiment].cpu().detach().numpy().item()
                    if sent_porb < 0.7:
                        break

                """auto-encoder loss & traning"""
                # training using discriminator loss
                enc_out = genmodel.encoder(del_idx)
                enc_out = enc_out.clone()
                token_idx = token_idx.clone()
                attribute = attribute.clone()
                dec_out, vocab_out = genmodel.decoder(enc_out, token_idx, attribute)
                # dec_out = dec_out.clone()
                vocab_out = vocab_out.clone()

                ## calculation loss
                recon_loss = genmodel.recon_loss(token_idx, vocab_out)
                # summary.add_scalar('reconstruction loss', recon_loss.item(), start)

                aed_trainer.zero_grad()
                recon_loss.backward(retain_graph=True) # retain_graph=True
                grad_norm = torch.nn.utils.clip_grad_norm_(genmodel.aed_params, max_grad_norm)
                aed_trainer.step()

                """decoder classification loss & training"""
                ## calculation loss
                # 수정함
                gen_cls_out = dismodel.gen_discriminator(vocab_out)

                ## calculation loss
                gen_cls_loss = genmodel.cls_loss(attribute, gen_cls_out)
                # summary.add_scalar('generated sentence loss', gen_cls_loss.item(), start)

                print("relu 수정17")
                import pdb; pdb.set_trace()

                gen_trainer.zero_grad()
                gen_cls_loss.backward() # retain_graph=True
                grad_norm = torch.nn.utils.clip_grad_norm_(genmodel.aed_params, max_grad_norm)
                gen_trainer.step()

        """savining point"""
        if (start+1)%pos_len == 0:
            random.shuffle(train_neg_dataset)
            random.shuffle(train_pos_dataset)
            save_model((start+1)//pos_len)
    save_model('final') # final_model


def save_model(iter):
    if not os.path.exists('models/'):
        os.makedirs('models/')
    torch.save(genmodel.state_dict(), 'models/gen_model_{}'.format(iter))

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()


from data import *
from define_distance import *
from prototype import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
        InputFeatures,
        AdamW,
        AutoConfig,
        AutoTokenizer,
        AutoModelForTokenClassification
        )
from spacy.lang.en import English 


model_name = 'clulab/roberta-timex-semeval'
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, config=config, use_fast=True)
model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)

model = model.cuda()

model.eval()

domain_transformer = nn.Linear(768, 768)
domain_transformer = domain_transformer.cuda()
domain_transformer.train()

alpha = 0.5 
optimizer = AdamW(domain_transformer.parameters(), lr=5e-5)

difdis = torch.load('/home/intern/bifrost/difdis_2')['model'].cuda() 
prototype_memory, num_prototype_, prototype_memory_dict, available_cls = proto_update(model)

len_cls = len(available_cls)

for i in range(3):
    for batch in target_train_dl:

        optimizer.zero_grad()

        input_ids = batch[0]
        attention_mask = batch[1]
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

        attention_mask_note = attention_mask.squeeze(0)
        attention_mask_note = torch.nonzero(attention_mask_note)
        attention_mask_note = len(attention_mask_note)

        fc1_lbcorr = model.roberta.forward(input_ids, attention_mask)[0].squeeze(0)
        fc1_lbcorr_list = fc1_lbcorr[:attention_mask_note]
        fc1_lbcorr_transformed = domain_transformer(fc1_lbcorr)
        fc1_lbcorr_transformed_list = fc1_lbcorr_transformed[:attention_mask_note]

        logits_before_transform = model.classifier.forward(fc1_lbcorr).view(-1,65)
        logits_after_transform = model.classifier.forward(fc1_lbcorr).view(-1,65)
        logits_after_transform_list = logits_after_transform[:attention_mask_note]

        m = nn.Softmax(dim=1)
        softmax_before = m(logits_before_transform)
        softmax_after = m(logits_after_transform)

        pseudo_label_origin = torch.argmax(softmax_before, dim=1)
        pseudo_label_origin_list = pseudo_label_origin[:attention_mask_note]

        entropy_before = torch.sum(-softmax_before*torch.log(softmax_before+1e-10), dim=1, keepdim=True) 
        entropy_before_norm = entropy_before / np.log(softmax_before.size(1)) 
        entropy_before_norm = entropy_before_norm.squeeze(1)
        entropy_before_norm_list = entropy_before_norm[:attention_mask_note]
        entropy_before_avg = torch.mean(entropy_before_norm_list)

        entropy_after = torch.sum(-softmax_after*torch.log(softmax_after+1e-10), dim=1, keepdim=True)
        entropy_after_norm = entropy_after / np.log(softmax_after.size(1))
        entropy_after_norm = entropy_after_norm.squeeze(1)
        entropy_after_norm_list = entropy_after_norm[:attention_mask_note]
        entropy_after_avg = torch.mean(entropy_after_norm_list)

        diss = torch.Tensor([]).cuda()
        for i in range(attention_mask_note):
            dis = difdis.forward(fc1_lbcorr_list[i], fc1_lbcorr_transformed_list[i])
            diss = torch.cat((diss, dis))
        
        distance_of_embed = torch.mean(diss)

        loss_1 = entropy_before_avg - entropy_after_avg
        loss_2 = distance_of_embed  
    
        total_loss = loss_1 + alpha*loss_2 

        total_loss.backward()
        optimizer.step()

torch.save({
    'model':domain_transformer
    }, './test_1')



import numpy as np 
from data import * 
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


model_name = 'clulab/roberta-timex-semeval'
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, config=config, use_fast=True)
fix_model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)  
train_model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)

fix_model.cuda()
trainable_cls = train_model.classifier
trainable_cls.train().cuda()

class DefinedDistance(nn.Module):
    def __init__(self):
        super(DefinedDistance, self).__init__()
        self.fc1 = trainable_cls 
        self.fc2 = nn.Linear(130, 1)

    def forward(self, vec_1, vec_2):
        cls_1 = F.relu(self.fc1(vec_1))
        cls_2 = F.relu(self.fc1(vec_2))
        cls_out = torch.cat((cls_1, cls_2))
        x = self.fc2(cls_out)
        return x 

difdis = DefinedDistance().cuda()
difdis.train()

optimizer = AdamW(difdis.parameters(), lr = 5e-5)

epoch = 3

for e in range(epoch):
    for batch in target_train_dl:
        optimizer.zero_grad()
        input_ids = batch[0]
        attention_mask = batch[1] 
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

        attention_mask_note = attention_mask.squeeze(0)
        attention_mask_note = torch.nonzero(attention_mask_note)
        attention_mask_note = len(attention_mask_note)

        fc1_lbcorr = fix_model.roberta.forward(input_ids, attention_mask)[0].squeeze(0)
        fc1_lbcorr_list = fc1_lbcorr[:attention_mask_note]
        logits = fix_model.classifier.forward(fc1_lbcorr).view(-1, 65)
        logits_list = logits[:attention_mask_note]
        m = nn.Softmax(dim=1)
        after_softmax = m(logits_list)
        pseudo_label = torch.argmax(after_softmax, dim=1)

        same_labels = torch.Tensor([]).cuda()
        diff_labels = torch.Tensor([]).cuda()

#0.1필터 추가할 것 
        entropy = torch.sum(-after_softmax*torch.log(after_softmax+1e-10), dim=1, keepdim=True)
        entropy_norm = entropy / np.log(after_softmax.size(1))
        entropy_norm = entropy_norm.squeeze(1)
        entropy_norm_list = entropy_norm[:attention_mask_note]

        over_threshold = (entropy_norm_list < 0.1).nonzero(as_tuple=True)[0] 
        fc1_lbcorr_list = fc1_lbcorr_list[over_threshold]

        for i in fc1_lbcorr_list:
            for j in fc1_lbcorr_list:
                if pseudo_label[(fc1_lbcorr == i).nonzero(as_tuple=True)[0][0]] == pseudo_label[(fc1_lbcorr == j).nonzero(as_tuple=True)[0][0]]:
                    same_labels = torch.cat((same_labels, difdis.forward(i, j)))
                else:
                    diff_labels = torch.cat((diff_labels, difdis.forward(i, j))) 

        loss_1 = torch.mean(same_labels)
        loss_2 = torch.mean(diff_labels)

        loss = loss_1 - loss_2
        loss.backward()
        optimizer.step()

torch.save({
    'model':difdis
    }, './difdis_2')

import anafora
import os 
import torch 
from torch.utils.data import Dataset, DataLoader
from transformers import (InputFeatures, AutoConfig, AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments)
from spacy.lang.en import English 

predict_dir = '/home/intern/bifrost/source-free-domain-adaptation/practice_text'
model_name = 'clulab/roberta-timex-semeval'
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, config=config, use_fast=True)
model = AutoModelForTokenClassification.from_pretrained(model_name, config=config) 

nlp = English()
nlp.add_pipe(nlp.create_pipe("sentencizer"))

source_classes = [i for i in range(65)]
target_classes = [i for i in range(65)]

class TimexInputFeatures(InputFeatures):

    def __init__(self, input_ids, attention_mask, offset_mapping):
        super().__init__(input_ids=input_ids, attention_mask=attention_mask)
        self.offset_mapping = offset_mapping 

    @classmethod
    def from_sentence(cls, input_data, sent_idx, sent_offset):
        input_ids = input_data["input_ids"][sent_idx]
        attention_mask = input_data["attention_mask"][sent_idx]
        offset_mapping = input_data["offset_mapping"][sent_idx]
        for token_idx, offset in enumerate(offset_mapping):
            start, end = offset.numpy()
            if start == end: 
                continue
            start += sent_offset
            end += sent_offset 
            offset_mapping[token_idx][0] = start 
            offset_mapping[token_idx][1] = end 
        return cls(
                input_ids, 
                attention_mask,
                offset_mapping 
                )

class TimexDataset(Dataset):

    def __init__(self, doc_indices, features):
        self.doc_indices = doc_indices
        self.features = features 

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return (self.features[i].input_ids, self.features[i].attention_mask)

    @classmethod 
    def from_texts(cls, text_dir, nlp, tokenizer):
        if not os.path.exists(text_dir):
            raise Exception("The %s directory does not exist." % text_dir)
        text_directory_files = anafora.walk(text_dir, xml_name_regex=".*((?<![.].{3})|[.]txt)$")
        features = []
        doc_indices = []
        for text_files in text_directory_files:
            doc_index = len(features)
            text_subdir_path, text_doc_name, text_file_names = text_files 
            if len(text_file_names) != 1:
                raise Exception("Wrong number of text files in %s" % text_subdir_path)
            text_file_path = os.path.join(text_dir, text_subdir_path, text_file_names[0])
            with open(text_file_path) as txt_file:
                text = txt_file.read()
            doc = nlp(text)
            input_raw = [sent.text_with_ws for sent in doc.sents]
            input_data = tokenizer(input_raw,
                                  return_tensors='pt',
                                  padding='max_length',
                                  truncation='longest_first',
                                  return_offsets_mapping=True
                    )
            sent_offset = 0 
            for sent_idx, _ in enumerate(input_data["input_ids"]):
                features.append(TimexInputFeatures.from_sentence(
                    input_data,
                    sent_idx,
                    sent_offset))
                sent_offset += len(input_raw[sent_idx])
            doc_indices.append((text_subdir_path, doc_index, len(features)))
        return cls(doc_indices, features)


dataset = TimexDataset.from_texts(predict_dir, nlp, tokenizer)
target_train_dl = DataLoader(dataset = dataset, batch_size=1)

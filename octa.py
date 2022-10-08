from transformers import PreTrainedTokenizerFast, RobertaModel
import os
import torch
import numpy as np
from torch import nn

model_dir = "C:/Users/hplis/PycharmProjects/roberta/roberta_base_transformers/"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))
tokenizer.pad_token = 0

# Valence_M
class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels_valence = df['norm_valence'].values.astype(float)
        self.labels_arousal = df['norm_arousal'].values.astype(float)
        self.labels_dominance = df['norm_dominance'].values.astype(float)
        self.labels_origin = df['norm_origin'].values.astype(float)
        self.labels_significance = df['norm_significance'].values.astype(float)
        self.labels_concretness = df['norm_concretness'].values.astype(float)
        self.labels_imegability = df['norm_imegability'].values.astype(float)
        self.labels_age_of_aquisition = df['norm_age_of_aquisition'].values.astype(float)

        self.labels_dominance_sd = df['norm_dominance_sd'].values.astype(float)
        self.labels_valence_sd = df['norm_valence_sd'].values.astype(float)
        self.labels_arousal_sd = df['norm_arousal_sd'].values.astype(float)
        self.labels_origin_sd = df['norm_origin_sd'].values.astype(float)
        self.labels_significance_sd = df['norm_significance_sd'].values.astype(float)
        self.labels_concretness_sd = df['norm_concretness_sd'].values.astype(float)
        self.labels_imegability_sd = df['norm_imegability_sd'].values.astype(float)
        self.labels_age_of_aquisition_sd = df['norm_age_of_aquisition_sd'].values.astype(float)

        self.texts = [tokenizer(str(text),
                               padding='max_length', max_length = 10, truncation=True,
                                return_tensors="pt") for text in df['polish word']]

    def classes(self):
        return self.labels_valence, self.labels_arousal, self.labels_dominance, self.labels_origin, self.labels_significance, self.labels_concretness, self.labels_imegability, self.labels_age_of_aquisition, self.labels_valencesd, self.labels_arousal_sd, self.labels_dominance_sd, self.labels_origin_sd, self.labels_significance_sd, self.labels_concretness_sd, self.labels_imegability_sd, self.labels_age_of_aquisition_sd

    def __len__(self):
        return len(self.labels_valence)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels_valence[idx]), np.array(self.labels_arousal[idx]), np.array(self.labels_dominance[idx]), np.array(self.labels_origin[idx]), np.array(self.labels_significance[idx]), np.array(self.labels_concretness[idx]), np.array(self.labels_imegability[idx]), np.array(self.labels_age_of_aquisition[idx]), np.array(self.labels_valence_sd[idx]), np.array(self.labels_arousal_sd[idx]), np.array(self.labels_dominance_sd[idx]), np.array(self.labels_origin_sd[idx]), np.array(self.labels_significance_sd[idx]), np.array(self.labels_concretness_sd[idx]), np.array(self.labels_imegability_sd[idx]), np.array(self.labels_age_of_aquisition_sd[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y



class BertRegression(nn.Module):

    def __init__(self, dropout=0.2, hidden_dim=768):

        super(BertRegression, self).__init__()

        self.bert = RobertaModel.from_pretrained(model_dir)
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.affect = nn.Linear(hidden_dim, 1)
        self.arousal = nn.Linear(hidden_dim, 1)
        self.dominance = nn.Linear(hidden_dim, 1)
        self.origin = nn.Linear(hidden_dim, 1)
        self.significance = nn.Linear(hidden_dim, 1)
        self.concreteness = nn.Linear(hidden_dim, 1)
        self.imageability = nn.Linear(hidden_dim, 1)
        self.aqcuisition = nn.Linear(hidden_dim, 1)

        self.af_sd = nn.Linear(hidden_dim, 1)
        self.ar_sd = nn.Linear(hidden_dim, 1)
        self.do_sd = nn.Linear(hidden_dim, 1)
        self.or_sd = nn.Linear(hidden_dim, 1)
        self.si_sd = nn.Linear(hidden_dim, 1)
        self.co_sd = nn.Linear(hidden_dim, 1)
        self.im_sd = nn.Linear(hidden_dim, 1)
        self.aq_sd = nn.Linear(hidden_dim, 1)

        self.l_1_affect = nn.Linear(hidden_dim, hidden_dim)
        self.l_1_arousal = nn.Linear(hidden_dim, hidden_dim)
        self.l_1_dominance = nn.Linear(hidden_dim, hidden_dim)
        self.l_1_origin = nn.Linear(hidden_dim, hidden_dim)
        self.l_1_significance = nn.Linear(hidden_dim, hidden_dim)
        self.l_1_concreteness = nn.Linear(hidden_dim, hidden_dim)
        self.l_1_imageability = nn.Linear(hidden_dim, hidden_dim)
        self.l_1_aqcuisition = nn.Linear(hidden_dim, hidden_dim)



        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_id, mask):

        _, x = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        x = self.dropout(x)
        x = self.l1(x) + x
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.l2(x) + x
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.l3(x) + x
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.relu(x)

        affect_all = self.relu(self.dropout(self.layer_norm(self.l_1_affect(x) + x)))
        arousal_all = self.relu(self.dropout(self.layer_norm(self.l_1_arousal(x) + x)))
        dominance_all = self.relu(self.dropout(self.layer_norm(self.l_1_dominance(x) + x)))
        origin_all = self.relu(self.dropout(self.layer_norm(self.l_1_origin(x) + x)))
        significance_all = self.relu(self.dropout(self.layer_norm(self.l_1_significance(x) + x)))
        concreteness_all = self.relu(self.dropout(self.layer_norm(self.l_1_concreteness(x) + x)))
        imageability_all = self.relu(self.dropout(self.layer_norm(self.l_1_imageability(x) + x)))
        aqcuisition_all = self.relu(self.dropout(self.layer_norm(self.l_1_aqcuisition(x) + x)))



        affect = self.sigmoid(self.affect(affect_all))
        arousal = self.sigmoid(self.arousal(arousal_all))
        dominance = self.sigmoid(self.dominance(dominance_all))
        origin = self.sigmoid(self.origin(origin_all))
        significance = self.sigmoid(self.significance(significance_all))
        concreteness = self.sigmoid(self.concreteness(concreteness_all))
        imageability = self.sigmoid(self.imageability(imageability_all))
        aqcuisition = self.sigmoid(self.aqcuisition(aqcuisition_all))



        affect_sd = self.sigmoid(self.af_sd(affect_all))
        arousal_sd = self.sigmoid(self.ar_sd(arousal_all))
        dominance_sd = self.sigmoid(self.do_sd(dominance_all))
        origin_sd = self.sigmoid(self.or_sd(origin_all))
        significance_sd = self.sigmoid(self.si_sd(significance_all))
        concreteness_sd = self.sigmoid(self.co_sd(concreteness_all))
        imageability_sd = self.sigmoid(self.im_sd(imageability_all))
        aqcuisition_sd = self.sigmoid(self.aq_sd(aqcuisition_all))



        return affect, arousal, dominance, origin, significance, concreteness, imageability, aqcuisition, affect_sd, arousal_sd, dominance_sd, origin_sd, significance_sd, concreteness_sd, imageability_sd, aqcuisition_sd


# add pad token



# use_cuda = torch.cuda.is_available()
#
# if use_cuda:
#     model = model.cuda()

class Dataset(torch.utils.data.Dataset):

    def __init__(self, words):


        self.texts = [tokenizer(str(text),
                               padding='max_length', max_length = 10, truncation=True,
                                return_tensors="pt") for text in words]

    def classes(self):
        return self.labels_valence, self.labels_arousal, self.labels_dominance, self.labels_origin, self.labels_significance, self.labels_concretness, self.labels_imegability, self.labels_age_of_aquisition, self.labels_valencesd, self.labels_arousal_sd, self.labels_dominance_sd, self.labels_origin_sd, self.labels_significance_sd, self.labels_concretness_sd, self.labels_imegability_sd, self.labels_age_of_aquisition_sd

    def __len__(self):
        return len(self.texts)

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)

        return batch_texts



def check_and_predict(word, model, show=False):
    word_tok = tokenizer(word,
                         padding='max_length', max_length=10, truncation=True,
                         return_tensors="pt")
    model = model.cuda()
    mask = word_tok['attention_mask'].to(device)
    input_id = word_tok['input_ids'].squeeze(1).to(device)
    o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15, o16 = model(input_id, mask)
    if show == True:
        print('valence: ' + str(o1.float().cpu().detach().numpy()))
        print('arousal: ' + str(o2.float().cpu().detach().numpy()))
        print('dominance: ' + str(o3.float().cpu().detach().numpy()))
        print('origin: ' + str(o4.float().cpu().detach().numpy()))
        print('significance: ' + str(o5.float().cpu().detach().numpy()))
        print('concretness: ' + str(o6.float().cpu().detach().numpy()))
        print('imageability: ' + str(o7.float().cpu().detach().numpy()))
        print('age of acquisition: ' + str(o8.float().cpu().detach().numpy()))

    return o1.float().cpu().detach().numpy(), o2.float().cpu().detach().numpy(), o3.float().cpu().detach().numpy(), o4.float().cpu().detach().numpy(), o5.float().cpu().detach().numpy(), o6.float().cpu().detach().numpy(), o7.float().cpu().detach().numpy(), o8.float().cpu().detach().numpy()

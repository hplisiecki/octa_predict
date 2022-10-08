import pandas as pd
from octa import BertRegression, check_and_predict, Dataset
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
import os
import torch
import pickle
from bs4 import BeautifulSoup
from gensim import corpora
from gensim.models import TfidfModel
from gensim.utils import simple_preprocess


corpus = pd.read_csv('C:\\Users\\hplis\\PycharmProjects\\coherence\\data\\data_11_23(1).csv')

# ########################################################################################################################


# n_s = []
# for n, s in tqdm(zip(corpus['name'], corpus['surname'])):
#     if (n,s) not in n_s:
#         n_s.append((n,s))
#
# txts_ns = []
# for i in tqdm(n_s):
#     current = corpus[(corpus['name'] == i[0]) & (corpus['surname'] == i[1])]
#     txt = ''
#     for j in current['text']:
#         txt += j
#     txts_ns.append(txt)
#
# names_and_surnames = pd.DataFrame(n_s, columns=['name', 'surname'])
# names_and_surnames['text'] = txts_ns
# # # save
# # names_and_surnames.to_csv('names_and_surnames.csv', index=False)
# # load
# names_and_surnames = pd.read_csv('names_and_surnames.csv')
#
# tokenized_documents = [simple_preprocess(doc) for doc in txts_ns]
# dictionary = corpora.Dictionary()
# # create a bag of words for each document
# bow = [dictionary.doc2bow(doc, allow_update=True) for doc in tokenized_documents]
# # create a tf-idf model
# tfidf_scores = TfidfModel(bow, smartirs='ntc')
# save both
# dictionary.save('dictionary.dict')
# tfidf_scores.save('tfidf_scores.tfidf')
# ########################################################################################################################



# morph = pd.read_csv('C:\\Users\\hplis\\PycharmProjects\\coherence\\morph_dataframe_clean.csv')
# adjectives = morph[morph['val'] == 'adj']
# from gensim import corpora
# a = [word for word in set(separate_words__) if word not in morph]
# dictionary = corpora.Dictionary(doc.split() for doc in corpus['text'])
# check frequency of the word 'ale'

# save dictionary
# with open('main_dictionary.pickle', 'wb') as handle:
#     pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load
# with open('main_dictionary.pickle', 'rb') as handle:
#     dictionary = pickle.load(handle)

class model_predict():
    def __init__(self):
        self.model = BertRegression().cuda()
        self.model.load_state_dict(torch.load('C:\\Users\\hplis\\PycharmProjects\\social_ai\\models\\roberta_octa.pt'))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def predict(self, words):
        test = Dataset(words)
        test_dataloader = torch.utils.data.DataLoader(test, batch_size=400)

        all = torch.zeros(1,8)
        curr = 0
        for test_input in test_dataloader:
            mask = test_input['attention_mask'].to(self.device)
            input_id = test_input['input_ids'].squeeze(1).to(self.device)

            o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15, o16 = self.model(input_id, mask)
            # detach
            o1, o2, o3, o4, o5, o6, o7, o8 = o1.detach().cpu(), o2.detach().cpu(), o3.detach().cpu(), o4.detach().cpu(), o5.detach().cpu(), o6.detach().cpu(), o7.detach().cpu(), o8.detach().cpu()
            if all.shape == torch.zeros(1,8).shape:
                all = torch.cat((o1, o2, o3, o4, o5, o6, o7, o8), dim=1)
            else:
                all = torch.cat((all, torch.cat((o1, o2, o3, o4, o5, o6, o7, o8), dim=1)), dim=0)

            # print progress
            # print(f'{curr}/{len(test_dataloader)}')
            curr += 1

        return all

# avereage all
def predict_name(name):
    n = name.split(' ')[0]
    s = name.split(' ')[1]
    all_texts = names_and_surnames[(names_and_surnames['name'] == n) & (names_and_surnames['surname'] == s)]
    # get index
    index = names_and_surnames.index[(names_and_surnames['name'] == n) & (names_and_surnames['surname'] == s)].values[0]
    temponary = dict(zip([dictionary[number[0]] for number in tfidf_scores[bow][index]], [number[1] for number in tfidf_scores[bow][index]]))

    separate_words__ = tokenized_documents[index]
    # # remove numbers
    #

    print('+++++')
    # print(len(set(separate_words__)))
    # retained_words = [word for word in set(separate_words__) if dictionary.dfs[dictionary.token2id[word]] < len(dictionary.dfs) * 0.001]
    # print(len(retained_words))
    # retained_words = [word for word in set(separate_words__) if word in list(adjectives['word'].values)]
    # print(len(retained_words))
    print('------')



    # words = [word for txt in all_texts['clean'] for word in set(txt.split()) if word in retained_words]
    # words = [word for word in set(words) if word not in common]

    current_tfidf = [temponary[word] for word in separate_words__]

    words_count_tensor = torch.zeros(len(separate_words__), dtype=torch.float)
    for i, score in enumerate(current_tfidf):
        words_count_tensor[i] = score

    words_count_tensor = words_count_tensor.view(-1, 1)

    ########################################
    all = model.predict(separate_words__)

    # sum
    # inverse = 1 / words_count_tensor

    all = all * words_count_tensor
    # order from lowest to highest
    all = torch.cat([all, words_count_tensor], dim=1)
    # order from lowest to highest according to the first dimension
    result_all = []
    for i in range(8):
        a = all[all[:, i].sort()[1]]
        a = a[int(-len(all)*0.05):]
        sum_ = a.sum(dim = 0)
        result = float(sum_[i]) / float(sum_[8])
        result_all.append(result)


    # sum_ = torch.sum(words_count_tensor)
    # mean = all.sum(dim=0) / sum_
    ########################################
    # mean = all.mean(dim=0)

    return result_all

model = model_predict()

kaczo = predict_name('Jarosław Kaczyński')
biedro = predict_name('Robert Biedroń')
tusko = predict_name('Donald Tusk')
bosako = predict_name('Krzysztof Bosak')
brauno = predict_name('Grzegorz Braun')
korwi = predict_name('Janusz Korwin-Mikke')


import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
plt.hist(tusko[:,1].detach().numpy(), bins=100)
plt.show()
plt.hist(kaczo[:,1].detach().numpy(), bins=100)
plt.show()
plt.hist(biedro[:,1].detach().numpy(), bins=100)
plt.show()
plt.hist(brauno[:,1].detach().numpy(), bins=100)
plt.show()
plt.hist(bosako[:,1].detach().numpy(), bins=100)
plt.show()


# to float
kaczo = [round(float(i), 3) for i in kaczo]
biedro = [round(float(i), 3) for i in biedro]
tusko = [round(float(i), 3) for i in tusko]
bosako = [round(float(i), 3) for i in bosako]
brauno = [round(float(i), 3) for i in brauno]
korwi = [round(float(i), 3) for i in korwi]

new_df = pd.DataFrame(columns=['valence', 'arousal', 'dominance', 'origin', 'significance', 'concreteness', 'imegability', 'age_of_acquisition'], data = [kaczo, biedro, tusko, bosako, brauno, korwi])
new_df['names'] = ['Jarosław Kaczyński', 'Robert Biedroń', 'Donald Tusk', 'Krzysztof Bosak', 'Grzegorz Braun', 'Janusz Korwin-Mikke']



a = list(set(kaczo).intersection(biedro))
a = list(set(a).intersection(tusko))
a = list(set(a).intersection(bosako))
a = list(set(a).intersection(brauno))
common = a





# import matplotlib
# matplotlib.use('TKAgg')
# import matplotlib.pyplot as plt
# plt.hist(words_count_tensor.detach().numpy(), bins=100)

def name_in_time(name):


    all_texts = corpus[corpus['fullname'] == name]
    # get index
    tokenized_documents = [simple_preprocess(doc) for doc in all_texts['text']]
    dictionary = corpora.Dictionary()
    # create a bag of words for each document
    bow = [dictionary.doc2bow(doc, allow_update=True) for doc in tokenized_documents]
    # create a tf-idf model
    tfidf_scores = TfidfModel(bow, smartirs='ntc')

    all_scores = [[] for _ in range(8)]

    for idx in tqdm(range(len(tokenized_documents))):
        temponary = dict(zip([dictionary[number[0]] for number in tfidf_scores[bow][idx]],
                             [number[1] for number in tfidf_scores[bow][idx]]))

        separate_words__ = tokenized_documents[idx]
        current_tfidf = [temponary[word] for word in separate_words__]

        words_count_tensor = torch.zeros(len(separate_words__), dtype=torch.float)
        for i, score in enumerate(current_tfidf):
            words_count_tensor[i] = score

        words_count_tensor = words_count_tensor.view(-1, 1)

        ########################################
        all = model.predict(separate_words__)
        # sum
        # inverse = 1 / words_count_tensor

        all = all * words_count_tensor
        sum_ = torch.sum(words_count_tensor)
        mean = all.sum(dim=0) / sum_
        for i in range(len(mean)):
            all_scores[i].append(mean[i])
    for i in range(len(all_scores)):
        all_scores[i] = [float(x) for x in all_scores[i]]
    all_texts['valence'] = all_scores[0]
    all_texts['arousal'] = all_scores[1]
    all_texts['dominance'] = all_scores[2]
    all_texts['origin'] = all_scores[3]
    all_texts['significance'] = all_scores[4]
    all_texts['concreteness'] = all_scores[5]
    all_texts['imegability'] = all_scores[6]
    all_texts['age_of_acquisition'] = all_scores[7]

    return all_texts

model = model_predict()

changes = pd.read_csv('changes.csv')
# count names
names = changes['name'].value_counts()
j = changes[changes['name'] == 'Robert Tyszkiewicz']

for name in corpus['fullname'].unique():
    if 'Braun' in name:
        print(name)



current_name = 'Grzegorz Braun'
temp_name = name_in_time(current_name)

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

# save
# chosen.to_csv('po_viktoria/chosen.csv', index=False)
year = []
for i in temp_name['data']:
    year.append(str(i.split('-')[0] + '-' + i.split('-')[1]))

temp_name['year'] = year
# group by month
year = temp_name.groupby('year').mean()
from matplotlib.pyplot import figure

figure(figsize=(16, 8), dpi=80)
plt.plot(year.index, year['valence'].rolling(window=3).mean(), label='valence', color = 'red')
plt.plot(year.index, year['arousal'].rolling(window=3).mean(), label='arousal', color = 'green')
plt.plot(year.index, year['dominance'].rolling(window=3).mean(), label='dominance', color = 'blue')
plt.plot(year.index, year['origin'].rolling(window=3).mean(), label='origin', color = 'black', linestyle='-.')
plt.plot(year.index, year['significance'].rolling(window=3).mean(), label='significance', color = 'pink')
plt.plot(year.index, year['concreteness'].rolling(window=3).mean(), label='concreteness', color = 'orange')
plt.plot(year.index, year['imegability'].rolling(window=3).mean(), label='imegability', color = 'grey')
plt.plot(year.index, year['age_of_acquisition'].rolling(window=3).mean(), label='acquisition', color = 'brown', linestyle='--')
# plt.plot(year.index, year['valence'], label='valence', color = 'red')
# plt.plot(year.index, year['arousal'], label='arousal', color = 'green')
# plt.plot(year.index, year['dominance'], label='dominance', color = 'blue')
# plt.plot(year.index, year['origin'], label='origin', color = 'black', linestyle='-.')
# plt.plot(year.index, year['significance'], label='significance', color = 'pink')
# plt.plot(year.index, year['concreteness'], label='concreteness', color = 'orange')
# plt.plot(year.index, year['imegability'], label='imegability', color = 'grey')
# plt.plot(year.index, year['age_of_acquisition'], label='acquisition', color = 'brown', linestyle='--')
plt.xticks(rotation=40)
for idx, label in enumerate(plt.gca().xaxis.get_ticklabels()):
    if idx % 3 != 0:
        label.set_visible(False)
# changes_list = j['prev_date'].tolist()
# changes_list = [ch.split('-')[0] + '-' + ch.split('-')[1] for ch in changes_list]
# # find closest month
# compare_list = [int(''.join(t.split('-'))) for t in year.index]
# compared_list = [int(''.join(t.split('-'))) for t in changes_list]
#
# compared = [min(compare_list, key=lambda x:abs(x-t)) for t in compared_list]
# changes_list = [str(ch)[:4] + '-' + str(ch)[4:] for ch in compared]
#
#
# for i in changes_list:
#     plt.axvline(x = i, color = 'b')
plt.title(current_name)
plt.legend()
ax = plt.subplot(111)

ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
          ncol=5, fancybox=True, shadow=True)
axes = plt.gca()
axes.yaxis.grid()

plt.show()
# save
plt.savefig('graphs/names_in_time/' + current_name + '.png')

########################################################################################################################

def predict_club(name):
    all_texts = corpus[corpus['klub'] == name]
    print(len(all_texts))

    words = [word for txt in all_texts['text'] for word in txt.split()]
    # calculate term frequency
    words_count = {}
    for word in words:
        if word in words_count:
            words_count[word] += 1
        else:
            words_count[word] = 1

    # create tensor with word frequency
    words_count_tensor = torch.zeros(len(words), dtype=torch.float)
    for i, word in enumerate(words):
        words_count_tensor[i] = words_count[word]

    words_count_tensor = words_count_tensor.view(-1,1)
    words_count_tensor = words_count_tensor.to(torch.float)
    # normalize
    words_count_tensor = (words_count_tensor - min(words_count_tensor)) / (max(words_count_tensor) - min(words_count_tensor))
    words_count_tensor = words_count_tensor + 0.5
    words_count_tensor = torch.cat((words_count_tensor, words_count_tensor, words_count_tensor, words_count_tensor, words_count_tensor,words_count_tensor ,words_count_tensor ,words_count_tensor), dim=1)

    all = model.predict(words)

    # multiply by the inverse of the word frequency
    all = all * (1 / words_count_tensor)

    return all

model = model_predict()

pis = predict_club('PiS')
po = predict_club('PO')



def predict_name(name):
    all_texts = corpus[corpus['fullname'] == name]
    txts = []
    for filename in all_texts['file']:
        with open(
                os.path.join('C:\\Users\\hplis\\PycharmProjects\\coherence\\clean_texts_09_11\\', filename),
                encoding='utf8') as f:  # open in readonly mode
            html = BeautifulSoup(f, "html.parser")
            text = html.getText()
            text = ''.join(c for c in text if c.isalpha() or c.isspace())
            txts.append(text)
    all_texts['clean'] = txts
    # print(len(all_texts))

    separate_words__ = [word for txt in all_texts['clean'] for word in txt.split()]

    print('+++++')
    print(len(set(separate_words__)))
    # retained_words = [word for word in set(separate_words__) if dictionary.dfs[dictionary.token2id[word]] < len(dictionary.dfs) * 0.01]
    # print(len(retained_words))
    # retained_words = [word for word in set(separate_words__) if list(stemmer.stem([word], parser)[0][1].values())[0][0].split(':')[0] in ['adj']]
    new = []
    for word in tqdm(list(set(separate_words__))):
        try:
            if list(stemmer.stem([word], parser)[0][1].values())[0][0].split(':')[0] in ['adj']:
                new.append(word)
        except:
            print(word)
    print(len(new))
    print('------')
    retained_words = new


    words = [word for txt in all_texts['clean'] for word in txt.split() if word in retained_words]



    all = model.predict(words)
    # sum
    # inverse = 1 / words_count_tensor

    # all = all * inverse

    mean = all.mean(dim=0)

    return mean

from pyltp import Segmentor
from pyltp import Postagger
import os
import jieba
from count_words import get_counts

LTP_DATA_DIR = '/Users/sunlinghao/PycharmProjects/StanfordNLP/ltp_data_v3.4.0'
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
sents = '管道共建各参建方应明确管道段落的起吃点、建设路由、管孔数、管孔规格、段长、人（手）孔设置、材料和工期安排。'
seg = Segmentor()
seg.load(cws_model_path)
words = seg.segment(sents)
seg.release()
words_ls = list(words)
print(words_ls)
c = get_counts(words_ls)
print(c)
print(" ".join(jieba.cut(sents)))
pos = Postagger()
words = ['的']
pos.load(pos_model_path)
print(list(pos.postag(words)))


from pyltp import Segmentor
from pyltp import Postagger
import os
import jieba

LTP_DATA_DIR = '/Users/sunlinghao/PycharmProjects/StanfordNLP/ltp_data_v3.4.0'
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
sents = '基础的混凝土强度等级、配筋等应符合设计规定'
seg = Segmentor()
seg.load(cws_model_path)
words = seg.segment(sents)
seg.release()
print(" ".join(words))
print(" ".join(jieba.cut(sents)))
pos = Postagger()
words = ['的']
pos.load(pos_model_path)

print(list(pos.postag(words)))
print("master")


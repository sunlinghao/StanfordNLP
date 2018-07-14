from nltk.parse.corenlp import CoreNLPParser
from nltk.parse.corenlp import CoreNLPServer
from nltk.tokenize.stanford_segmenter import StanfordSegmenter
import os
import jieba
import googletrans
from nltk.tag import StanfordPOSTagger
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
os.environ['CLASSPATH'] = "/Users/sunlinghao/PycharmProjects/StanfordNLP/stanford-corenlp-full-2018-02-27"

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence(sentence):
    res = []
    lemmatizer = WordNetLemmatizer()
    for word, pos in pos_tag(word_tokenize(sentence)):
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
        res.append(lemmatizer.lemmatize(word, pos=wordnet_pos))

    return res


# with CoreNLPServer(port=9000) as server:
#     parser = CoreNLPParser()
#     t = parser.raw_parse("The introduction,in the book is a summary of what is in the text")
#     next(t).pretty_print()
# trans = googletrans.Translator()
# t = trans.translate("地").text
# print(t)
# res = lemmatize_sentence("I am a student.")
# print(res)
# lemmatizer = WordNetLemmatizer()
# print(lemmatizer.lemmatize('Dogs are animials'))
print(pos_tag(["he"]))
#
# os.environ['CLASSPATH'] = "/Users/sunlinghao/PycharmProjects/StanfordNLP/stanford-corenlp-full-2018-02-27:/Users/sunlinghao/PycharmProjects/StanfordNLP/stanford-postagger-full-2018-02-27"
#
# # os.environ['CLASSPATH'] = "/Users/sunlinghao/PycharmProjects/StanfordNLP/stanford-postagger-full-2018-02-27"
# # print(os.environ)
# os.environ['STANFORD_MODELS'] = "/Users/sunlinghao/PycharmProjects/StanfordNLP/stanford-postagger-full-2018-02-27/models"
#
# seg_list = jieba.cut("消防器材的设置应符合消防部门有关法规和国家现行有关标准的规定，并应定期进行检查、更新。")
# seg_list = list(seg_list)
# print(seg_list)
# # print(" ".join(seg_list))
#
# seg = []
# if not seg:
#     print("hello")
#
#
# chi_tagger = StanfordPOSTagger('chinese-distsim.tagger')
# for _, word_and_tag in  chi_tagger.tag(seg_list):
#     word, tag = word_and_tag.split('#')
#     print(word, tag)
# print(chi_tagger.tag(seg_list))
# with CoreNLPServer(port=9000) as server:
#
#     parser = CoreNLPParser()
#     print(list(parser.raw_parse('I am a student.'))[0])
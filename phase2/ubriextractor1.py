from nltk.parse.corenlp import CoreNLPParser
from nltk.parse.corenlp import CoreNLPServer
import os
import googletrans
import nltk
import queue
import jieba
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

os.environ['CLASSPATH'] = "/Users/sunlinghao/PycharmProjects/StanfordNLP/stanford-corenlp-full-2018-02-27"

class UBRINlpExtractor:

    def __init__(self, sentence):
        with CoreNLPServer(port=9000) as server:
            en_parser = CoreNLPParser()
            # sg = StanfordTokenizer(path_to_jar='../stanford-parser-full-2018-02-27/stanford-parser.jar')
            self.trans = googletrans.Translator()

            self.sentence = sentence

            result1 = self.trans.translate(sentence).text
            print(result1)
            # en_sencence = result1.split(".")
            # print(en_sencence)
            # tree = list(en_parser.raw_parse(result1))
            iter = en_parser.raw_parse_sents([result1])
            tree = []
            while True:
                try:
                     sub_tree = list(next(iter))
                     tree.append(sub_tree)
                except StopIteration:
                    break
            print(len(tree))
            self.tree = tree[0][0]
            self.rel=[]

    def first_minNP(self,node=None):
        """
        extract the first Minimal NP
        :param node:
        :return:nltk tree node
        """
        if(node == None):
            node = self.tree
        result = None
        for child in node:
            if isinstance(child, nltk.tree.Tree):
                if child.label() == "NP":
                    # 将np节点赋予结果
                    result = child
                tmp_result = self.first_minNP(child)
                # 存在更小的np
                if tmp_result:
                    result = tmp_result
            if result:
                break
        return result

    def BFSFirstNP(self,node = None):
        """
        Breadth first searching find the first NP in the node by BFS
        :return:
        """
        if node == None:
            node = self.tree
        q = queue.Queue()
        q.put(node)
        while not q.empty():

            now_node = q.get_nowait()

            for child in now_node:
                if isinstance(child, nltk.tree.Tree):
                    if child.label() == "NP":
                        return child
                    else:
                        q.put(child)

    def find_entity_from_PP(self, node=None):
        """
        This function uses to extract the entity, if the entity in the node.
        Extract information from a PP node in the given node.
        :param node: firstNP node
        :return:
        """
        if node is None:
            # find BFSFirstNP in sentence
            node = self.BFSFirstNP()
        isFind = False
        NPNode = nltk.tree.Tree
        q = queue.Queue()
        q.put(node)
        while not q.empty():
            now_node = q.get_nowait()
            for child in now_node:
                if isinstance(child, nltk.tree.Tree):
                    q.put(child)
                    if child.label() == "PP":
                        NPNode = child
                        isFind = True
        if isFind:
            for child in NPNode:
                if child.label() == "NP":
                    return child
            return NPNode
        else:
            return node

    def get_trans(self, node):
        result2 = self.trans.translate(" ".join(node.leaves()), dest="zh-cn", src='en')
        return result2.text

    def drawTree(self):
        self.tree.draw()

    def get_original_entity(self):
        seg = jieba.cut(self.sentence)
        seg = list(seg)
        en_entity = self.find_entity_from_PP()
        # 转换为小写
        en_entity_list = " ".join(en_entity.leaves()).lower().split()

        # 找到lemma
        lemma = WordNetLemmatizer()
        temp_lemma_list = []
        for word in en_entity_list:
            word_lemma = lemma.lemmatize(word)
            temp_lemma_list.append(word_lemma)
        en_entity_str = " ".join(temp_lemma_list)
        en_entity_list = temp_lemma_list
        print(en_entity_str)

        # 初始化所需数据
        entity = []
        location = -1
        zh_index = -1
        # 短语匹配
        for zh_phrase in seg:
            # 当前中文单词的位置
            location += 1
            # 翻译成英文
            en_phrase = self.trans.translate(zh_phrase).text.lower()
            # 单词匹配
            for word in en_phrase.split():
                # lemma
                word = lemma.lemmatize(word)
                # 判断词性
                pos = pos_tag([word])
                # 词性为CC 或者 IN 不作为判断词 # 让虚词不参与
                if pos[0][1] == 'CC' or pos[0][1] == 'IN':
                    continue

                # 限定词语出现位置
                en_entity_domain = en_entity_list[:location+1]



                # 得到单词
                if word in en_entity_domain:
                    en_index = 0
                    # 得到单词在英文实体中的位置
                    # while en_entity_list[en_index]!=word:
                    #     # 不计算冠词
                    #     if pos_tag([en_entity_list[en_index]])[0][1] != 'DT':
                    #         en_index += 1

                    en_index = en_entity_list.index(word)

                    process_index = en_index
                    # 英文单词数多于中文时，保证不会选到之前的中文
                    if en_index > location - zh_index - 1:
                        process_index = location - zh_index - 1

                    # delete_count += 1
                    # is_find = True
                    # 得到匹配单词的位置

                    # 加入之前未识别单词
                    temp_en_index = 0
                    while process_index > 0:

                        if pos_tag([en_entity_list[temp_en_index]])[0][1] != 'DT':
                            # 遇到英文冠词，略过（主要是the）
                            entity.append(seg[location - process_index])
                        temp_en_index += 1
                        print(temp_en_index)
                        process_index -= 1
                    # process_index<0 说明一个中文词分为多个英文词，不必重复添加中文词
                    if process_index == 0:
                        entity.append(zh_phrase)
                    # 删除已经匹配的entity
                    en_entity_list = en_entity_list[en_index + 1:]
                    en_entity_str = " ".join(en_entity_list)
                    zh_index = location

        if en_entity_str != "":
            print("do not deal all"+en_entity_str)

        if en_entity_str == "":
            print('finish')

        return "".join(entity)

    def get_clause_node(self):
        """
                Breadth first searching find the first S in the node by BFS
                :return:
                """
        node = self.tree
        q = queue.Queue()
        q.put(node)
        while not q.empty():

            now_node = q.get_nowait()

            for child in now_node:
                if isinstance(child, nltk.tree.Tree):
                    if child.label() == "SBAR":
                        return child
                    else:
                        q.put(child)





if __name__ == '__main__':
    # s = "为了加强城乡规划管理，协调城乡空间布局，改善人居环境，促进城乡经济社会全面协调可持续发展，制定本法。"
    # s = "运行管理、操作和维护人员应定期培训。"
    f_input= open("result_10.txt", encoding='utf-8')
    f_output = open("chinese.txt", "w+", encoding='utf-8')
    sentence_index = 0
    while sentence_index<10:
        sentence_index += 1
        s = f_input.readline()
        s = "当管道损坏时，应进行维修。当管道良好时，不需要维修。"
        test = UBRINlpExtractor(s)
        final_entity = test.get_original_entity()
        test.drawTree()

        # seg = jieba.cut(s)
        # seg = list(seg)
        #

        # en_entity = test.find_entity_from_PP()
        # # 转换为小写
        # en_entity_str = " ".join(en_entity.leaves()).lower()
        # # print(en_entity_str)
        # # # 中文
        # # ttt = test.get_trans(en_entity).lower()
        #
        #
        # en_entity_list = en_entity_str.split()
        # # 找到lemma
        # lemma = WordNetLemmatizer()
        # temp_lemma_list = []
        # for word in en_entity_list:
        #     word_lemma = lemma.lemmatize(word)
        #     temp_lemma_list.append(word_lemma)
        # en_entity_str = " ".join(temp_lemma_list)
        # en_entity_list = temp_lemma_list
        # # for item in en_entity_list:
        # #     if en_entity_list.count(item)>1:
        # #         print('无法处理')
        #
        #
        #
        #
        # print(en_entity_str)
        #
        # entity = []
        # trans = googletrans.Translator()
        # location = -1
        # zh_index = -1
        # # 短语匹配
        # for zh_phrase in seg:
        #     # 当前中文单词的位置
        #     location += 1
        #     # 翻译成英文
        #     en_phrase = trans.translate(zh_phrase).text.lower()
        #     # 单词匹配
        #     for word in en_phrase.split():
        #         # lemma
        #         word = lemma.lemmatize(word)
        #         # 限定词语出现位置
        #         # en_entity_domain = en_entity_list[:location+1]
        #
        #         # 得到单词
        #         if word in en_entity_list:
        #             # 得到单词在英文实体中的位置
        #             en_index = en_entity_list.index(word)
        #             process_index = en_index
        #             # 英文单词数多于中文时，保证不会选到之前的中文
        #             if en_index > location - zh_index - 1:
        #                 process_index = location - zh_index - 1
        #
        #
        #             # delete_count += 1
        #             # is_find = True
        #             # 得到匹配单词的位置
        #
        #             # 加入之前未识别单词
        #             while process_index > 0:
        #
        #                 entity.append(seg[location-process_index])
        #                 process_index -= 1
        #             # process_index<0 说明一个中文词分为多个英文词，不必重复添加中文词
        #             if process_index == 0:
        #                 entity.append(zh_phrase)
        #             # 删除已经匹配的entity
        #             en_entity_list = en_entity_list[en_index+1:]
        #             en_entity_str = " ".join(en_entity_list)
        #             zh_index = location






        # 单词匹配
        # if len(entity) == 0:
        #     print("翻译变化")
        #     for it in seg:
        #         # 翻译中文
        #         it_trans = trans.translate(it).text.lower()
        #         # 将短语拆开
        #         for i in it_trans.split():
        #             # 单词在之前实体中
        #             if i in str_result_list:
        #                 core_index = str_result_list.index(i)
        #                 print(core_index)
        #                 str_result_list = str_result_list[core_index+1:]
        #                 str_result = " ".join(str_result_list)
        #                 entity.append(it)
        #                 break

        # test.tree.draw()
        # if en_entity_str != "":
        #     print(en_entity_str)
        # final_entity = "".join(entity)
        print(final_entity)
        f_output.write(final_entity+"\n")
        if final_entity in s:
            print(sentence_index)
        # t = trans.translate(str_result,dest="zh-cn")
        # print(t)

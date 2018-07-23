from nltk.parse.corenlp import CoreNLPParser
from nltk.parse.corenlp import CoreNLPServer
import os
import googletrans
import nltk
import queue
import jieba
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.parse.stanford import StanfordParser
from nltk.tokenize.stanford import StanfordTokenizer
from pyltp import Segmentor


LTP_DATA_DIR = '/Users/sunlinghao/PycharmProjects/StanfordNLP/ltp_data_v3.4.0'
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')


class UBRINlpExtractor:

    def __init__(self, sentence):
        en_parser = StanfordParser(path_to_jar='../stanford-parser-full-2018-02-27/stanford-parser.jar',
                                   path_to_models_jar='../stanford-parser-full-2018-02-27/stanford-parser-3.9.1-models.jar',
                                   model_path='../stanford-parser-full-2018-02-27/stanford-parser-3.9.1-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')
        sg = StanfordTokenizer(path_to_jar='../stanford-parser-full-2018-02-27/stanford-parser.jar')
        self.status = 0
        self.trans = googletrans.Translator()

        self.sentence = sentence.strip("\n").replace(" ", "")

        en_trans = self.trans.translate(sentence).text
        en_trans = sg.tokenize(en_trans)
        try:
            tree = list(en_parser.parse(en_trans))
            self.tree = tree[0]
            # print(self.tree)
            self.rel = []
        except:
            self.status = 1


        # with CoreNLPServer(port=9000) as server:
        #     self.status = 0
        #     en_parser = CoreNLPParser()
        #     # sg = StanfordTokenizer(path_to_jar='../stanford-parser-full-2018-02-27/stanford-parser.jar')
        #     self.trans = googletrans.Translator()
        #
        #     self.sentence = sentence.strip("\n").replace(" ","")
        #     en_trans = self.trans.translate(sentence).text
        #     try:
        #         tree = list(en_parser.raw_parse(en_trans))
        #         self.tree = tree[0]
        #         # print(self.tree)
        #         self.rel=[]
        #     except:
        #         self.status = 1

            # tree = list(en_parser.raw_parse_sents(en_trans_list))


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
        return node

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
        else:
            node = self.BFSFirstNP(node)

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
        # 得到英文实体
        en_entity = self.find_entity_from_PP()
        # 转换为小写
        en_entity_list = " ".join(en_entity.leaves()).lower().split()
        # Lemmatizer
        lemma = WordNetLemmatizer()
        temp_lemma_list = []
        for word in en_entity_list:
            word_lemma = lemma.lemmatize(word)
            temp_lemma_list.append(word_lemma)
        en_entity_str = " ".join(temp_lemma_list)
        en_entity_list = temp_lemma_list

        if pos_tag([en_entity_str])[0][1] == "PRP":
            print("指代消解")
            clause_node = self.get_clause_node()
            S_node = self.get_S(clause_node)
            if S_node is None:
                S_node = clause_node
            np_node = self.find_entity_from_PP(S_node)
            print(np_node.leaves())
            # 转换为小写
            en_entity_list = " ".join(np_node.leaves()).lower().split()
            # Lemmatizer
            temp_lemma_list = []
            for word in en_entity_list:
                word_lemma = lemma.lemmatize(word)
                temp_lemma_list.append(word_lemma)
            en_entity_str = " ".join(temp_lemma_list)
            en_entity_list = temp_lemma_list

        # 处理英文冠词在开头
        # dt = False
        # dt_index = seg_index
        pos_deal = pos_tag(en_entity_list[0:1])
        if pos_deal[0][1] == 'DT':
            en_entity_list = en_entity_list[1:]
            en_entity_str = " ".join(en_entity_list)

        # 中文分句
        seg = Segmentor()
        seg.load(cws_model_path)
        chi_seg = list(seg.segment(self.sentence))
        seg.release()
        # chi_seg = jieba.cut(self.sentence)
        # chi_seg = list(chi_seg)

        # 逐词翻译
        en_seg = []
        for chi_phrase in chi_seg:
            en_phrase = self.trans.translate(chi_phrase, src='zh-cn').text
            zh_word_lemma=[]
            for word in en_phrase.split():
                word = lemma.lemmatize(word.lower())
                zh_word_lemma.append(word)
            zh_word_lemma = " ".join(zh_word_lemma)
            en_seg.append(zh_word_lemma)
        # print(en_seg)
        # 标注翻译后的英文词性
        en_pos = pos_tag(en_seg)

        entity_list = []
        # 直接能在分次后翻译的句子中找到实体原词
        en_seg_sentence = " ".join(en_seg)
        if en_entity_str in en_seg_sentence:
            for word in en_entity_list:
                entity_list.append(chi_seg[en_seg.index(word)])
            return "".join(entity_list)

        zh_index = 0
        last_word_index = -1
        for phrase in zip(chi_seg,en_pos):
            trans_ls = phrase[1][0].split()

            is_add = False
            for word in trans_ls:
                # 连词和介词出现次数太多，不做评价标准
                # 放在此处可以让zh_index正常计数
                if phrase[1][1] == 'CC' or phrase[1][1] == 'IN':
                    continue
                # 不太需要  连词过多  无法划分时需要
                # if pos_tag([word])[0][1] == 'CC':
                #     continue

                # 限定词语出现在实体中的位置  （翻译的时候不会主语后置）
                en_entity_domain = en_entity_list[:zh_index+5]

                if word in en_entity_domain:
                    # 得到英文翻译在英文实体中的位置
                    en_index = en_entity_list.index(word)
                    # 处理大词翻译前面没有对上的情况  eg Thermal storage method / heat storage method  第二个词才对上，不会向前寻找
                    if not is_add:
                        # trans_ls 中文翻译列表
                        # seg_index 翻译后的该词的正确位置
                        seg_index = trans_ls.index(word)
                        seg_index = en_index - seg_index
                        seg_index = seg_index if seg_index > 0 else 0
                    else:
                        seg_index = en_index

                    # 英文单词数多于中文时，保证不会选到之前的中文
                    # if en_index > zh_index - last_word_index - 1:
                    secure_index = zh_index - last_word_index - 1

                    # 开头冠词放到前面处理

                    process_index = min(secure_index, seg_index)
                    # 加入之前未识别单词
                    # "的" 的影响只能添加一次
                    has_patch = False
                    while process_index > 0:

                        if secure_index > seg_index and (not has_patch):
                            # 中文中的介词在英文中没有翻译（"基础的混凝土强度"， "的"没有翻译）  有没有可能重复？
                            # print(pos_tag())
                            if en_pos[zh_index - process_index][1] == 'IN' and en_pos[zh_index - seg_index - 1][1] != "IN":
                                # print(en_pos[zh_index - seg_index - 1])
                                entity_list.insert(0, chi_seg[zh_index - seg_index - 1])
                                has_patch = True

                        entity_list.append(chi_seg[zh_index - process_index])

                        last_word_index += 1

                        process_index -= 1
                    # process_index<0 说明一个中文词分为多个英文词，不必重复添加中文词
                    if process_index == 0:
                        # 中文中实词前的介词在英文中没有翻译，例如“等”、"各"  且 en_index = 0
                        if (en_pos[zh_index-1][1] == 'IN' or en_pos[zh_index-1][1] == 'DT') and en_index == 0 and entity_list and last_word_index != zh_index:
                            entity_list.append(chi_seg[zh_index-1])
                            last_word_index += 1
                        entity_list.append(phrase[0])
                        # 添加完单词，修改指针
                        if last_word_index + 1 != zh_index:
                            print("not equal")
                        last_word_index = zh_index
                        is_add = True
                    # 删除已经匹配的entity
                    en_entity_list = en_entity_list[en_index + 1:]
                    en_entity_str = " ".join(en_entity_list)

            # 中文单词位置 + 1
            zh_index += 1
            if not en_entity_list:
                break

        if en_entity_str != "":
            # print("do not deal all:" + en_entity_str)
            return "".join(entity_list)+en_entity_str

        if en_entity_str == "":
            print('finish')

        return "".join(entity_list)

    def get_clause_node(self):
        """
                Breadth first searching find the first SBAR in the node by BFS
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

    def get_S(self,node):
        """
                 Depth first searching find the first minimum S in the node by BFS
                 :return:
                 """
        if(node == None):
            node = self.tree
        result = None
        for child in node:
            if isinstance(child, nltk.tree.Tree):
                if child.label() == "S":
                    # 将np节点赋予结果
                    result = child
                tmp_result = self.get_S(child)
                # 存在更小的np
                if tmp_result:
                    result = tmp_result
            if result:
                break
        return result






if __name__ == '__main__':
    # s = '当对螺纹接头采用密封焊时，外露螺纹应全部密封焊。'
    s = '基础的混凝土强度等级、配筋等应符合设计规定。'
    # print(s)
    test = UBRINlpExtractor(s)
    node = test.find_entity_from_PP()
    # test.drawTree()
    result = test.get_trans(node)
    # print(result)
    print(test.get_original_entity())


    # fileList = os.listdir("E:/lab/task/tuple")
    # fileList.remove("CECS 177-2005 城市地下通信塑料管道工程施工及验收规范.txt")
    # fileList.remove("CJJ 101-2016 埋地塑料给水管道工程技术规程 .txt")
    # fileList.remove("CJJ 105-2005 城镇供热管网结构设计规范.txt")
    # fileList.remove('CJJ 120-2008 城镇排水系统电气与自动化工程技术规程.txt')
    # fileList.remove('CJJ 138-2010 城镇地热供热工程技术规程-一级.txt')
    # fileList.remove('CJJ 6-2009 城镇排水管道维护安全技术规程.txt')
    # fileList.remove('CJJ 28-2014 城镇供热管网工程施工及验收规范.txt')
    # fileList.remove('CJJ 33-2005 城镇燃气输配工程施工及验收规范.txt')
    # fileList.remove('CJJ 51-2006 城镇燃气设施运行、维护和抢修安全技术规程.txt')
    # fileList.remove('CJJ 58-1994 城镇供水厂运行、维护及安全技术规程 .txt')
    # fileList.remove('CJJ 68-2016 城镇排水管渠与泵站运行、维护及安全技术规程-一级.txt')
    # fileList.remove('CJJ 88-2014 城镇供热系统运行维护技术规程.txt')
    # fileList.remove('CJJ 92-2016 城镇供水管网漏损控制及评定标准.txt')
    # fileList.remove('CJJ 95-2013 城镇燃气埋地钢质管道腐蚀控制技术规程.txt')
    # fileList.remove('CJJ 140-2010 二次供水工程技术规程(非正式版).txt')
    # fileList.remove('CJJ 146-2011 城镇燃气报警控制系统技术规程.txt')
    # fileList.remove('CJJ 159-2011 城镇供水管网漏水探测技术规程-一级.txt')
    # fileList.remove('CJJ 181-2012 城镇排水管道检测与评估技术规程.txt')
    # fileList.remove('CJJ 207-2013 城镇供水管网运行、维护及安全技术规程-一级.txt')
    # fileList.remove('CJJ 215-2014 城镇燃气管网泄漏检测技术规程-一级.txt')
    # fileList.remove('CJJT 209-2013 塑料排水检查井应用技术规程.txt')
    # fileList.remove('CJJT 210-2014 城镇排水管道非开挖修复更新工程技术规程.txt')
    # fileList.remove('CJJT 226-2014 城镇供水管网抢修技术规程-一级.txt')
    # fileList.remove('CJJT 230-2015 排水工程混凝土模块砌体结构技术规程.txt')
    # fileList.remove('CJJT 241-2016 城镇供热监测与调控系统技术规程.txt')
    # fileList.remove('CJJT 244-2016 城镇给水管道非开挖修复更新工程技术规程-一级.txt')
    # fileList.remove('CJJT 254-2016 城镇供热直埋热水管道泄漏监测系统技术规程-一级.txt')
    # fileList.remove('CJJT 268-2017 城镇燃气工程智能化技术规范-一级.txt')
    # fileList.remove('CJJ／T 147-2010 城镇燃气管道非开挖修复更新工程技术规程-一级.txt')
    # fileList.remove('CJJ／T 216-2014 燃气热泵空调系统工程技术规程-一级.txt')
    # fileList.remove('CJJ／T 250-2016 城镇燃气管道穿跨越工程技术规程.txt')
    # fileList.remove('CJT200-2004  城镇供热预制直埋蒸汽保温管技术条件.txt')
    # fileList.remove('DGTJ08-308-2002 埋地塑料排水管道工程技术规程.txt')
    # fileList.remove('DL 5190.5-2012 电力建设施工技术规范 第5部分：管道及系统-一级.txt')
    # fileList.remove('DLT 1006-2006 架空输电线路巡检系统.txt')
    # fileList.remove('DLT5106-1999_跨越电力线路架线施工规程-一级.txt')
    # fileList.remove('DLT 5221-2016 城市电力电缆线路设计技术规定- 一级.txt')
    # fileList.remove('GB 15558.1-2003 燃气用埋地聚乙烯(PE)管道系统　第1部分：管材.txt')
    # fileList.remove('GB 17051-1997 二次供水设施卫生规范.txt')
    # fileList.remove('GB 26255.2-2010 燃气用聚乙烯管道系统的机械管件 第2部分：公称外径大于63mm的管材用钢塑转换管件-一级.txt')
    # fileList.remove('GB 50013-2006 室外给水设计规范-一级.txt')
    # fileList.remove('GB 50014-2006(2016年版) 室外排水设计规范-一级.txt')
    # fileList.remove('GB 50028-2006 城镇燃气设计规范-一级.txt')
    # fileList.remove('GB 50032-2003室外给水排水和燃气热力工程抗震设计规范.txt')
    # fileList.remove('GB 50069-2002给水排水工程构筑物结构设计规范.txt')
    # fileList.remove('GB 50141-2008 给水排水构筑物工程施工及验收规范(附条文说明)-一级.txt')
    # fileList.remove('GB 50184-2011 工业金属管道工程施工质量验收规范.txt')
    # fileList.remove('GB 50236-2011 现场设备、工业管道焊接工程施工规范-一级.txt')
    # fileList.remove('GB 50268-2008 给水排水管道工程施工及验收规范-一级.txt')
    # fileList.remove('GB 50282-2016 城市给水工程规划规范.txt')
    # fileList.remove('GB 50289-2016 城市工程管线综合规划规范.txt')
    # fileList.remove('GB 50293-2014 城市电力规划规范-一级.txt')
    # fileList.remove('GB 50318-2017 城市排水工程规划规范.txt')
    # fileList.remove('GB 50373-2006 通信管道与通道工程设计规范-一级.txt')
    # fileList.remove('GB 50494-2009 城镇燃气技术规范.txt')
    # fileList.remove('GB 50811-2012 燃气系统运行安全评价标准.txt')
    # fileList.remove('GB 51098-2015 城镇燃气规划规范.txt')
    # fileList.remove('GB 51171-2016 通信线路工程验收规范-一级.txt')
    #
    #
    #
    # for file in fileList:
    #     print(file)
    #     f_input = open("../data/"+file, encoding='utf-8')
    #     f_output = open(file, "w+", encoding='utf-8')
    #
    # # f_input = open("../data/CJJ 101-2016 埋地塑料给水管道工程技术规程 .txt", encoding='utf-8')
    # # f_output = open("CJJ 101-2016 埋地塑料给水管道工程技术规程 .txt", "w+", encoding='utf-8')
    #     sentence_index = 0
    #     while True:
    #     # while sentence_index < 200:
    #         sentence_index += 1
    #         s = f_input.readline()
    #         if not s:
    #             break
    #         if len(s)< 6 or len(s) > 200:
    #             print("Not in range")
    #             continue
    #         # s = "对构筑物、建筑物的结构及各种阀门、护栏、爬梯、管道、井盖、盖板、支架、栈桥和照明设备等应定期进行检查、维护和维修。"
    #         test = UBRINlpExtractor(s)
    #         if test.status == 0:
    #             final_entity = test.get_original_entity()
    #             if final_entity == "":
    #                 f_output.write(str(sentence_index)+"failed")
    #             # entity = test.find_entity_from_PP()
    #             # f_output.write(test.get_trans(entity)+"\n")
    #             # print(entity)
    #             f_output.write(s)
    #             f_output.write(final_entity+"\n")
    #             print(final_entity)
    #         else:
    #             f_log = open("error.txt", "a+", encoding='utf-8')
    #             f_log.write(file+" "+test.sentence)
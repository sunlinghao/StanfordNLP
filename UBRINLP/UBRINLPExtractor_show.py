from nltk.parse.stanford import StanfordParser
import queue
import nltk
import googletrans
from nltk.tokenize.stanford import StanfordTokenizer
from nltk.tokenize.stanford_segmenter import StanfordSegmenter

class UBRINlpExtractor:


    def __init__(self, sentence):

        en_parser = StanfordParser(path_to_jar='../stanford-corenlp-full-2018-02-27/stanford-corenlp-3.9.1.jar',
                                   path_to_models_jar='../stanford-corenlp-full-2018-02-27/stanford-corenlp-3.9.1-models.jar',
                                   )
        sg = StanfordTokenizer(path_to_jar='../stanford-corenlp-full-2018-02-27/stanford-corenlp-3.9.1.jar')
        self.trans = googletrans.Translator()

        self.sentence = sentence

        result1 = sg.tokenize(self.trans.translate(sentence).text)

        tree = list(en_parser.parse(result1))
        self.tree = tree[0]
        self.rel=[]

    def first_minNP(self,node=None):
        '''
        extract the first Minimal NP
        :param node:
        :param isFind:
        :return:nltk tree node
        '''
        if(node == None):
            node = self.tree
        result = None
        for child in node:
            if isinstance(child, nltk.tree.Tree):
                if child.label() == "NP":
                    # 将np节点赋予结果
                    result = child
                tmpresult = self.first_minNP(child)
                # 存在更小的np
                if tmpresult != None:
                    result = tmpresult
            if result:
                break
        return result

    def BFSFirstNP(self,node = None):
        '''
        Breadth first searching find the first NP in the node by BFS
        :return:
        '''
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

    def find_entity(self, node=None):
        '''
        This function uses to extract the entity, if the entity in the node.
        Extract information from a PP node in the given node
        :param node: firstNP node
        :return:
        '''
        if node == None:
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

    def firstVP(self,node = None):
        '''
        Breadth first searching find the first VP by BFS
        :return:
        '''
        if node == None:
            node = self.tree
        q = queue.Queue()
        q.put(node)
        while not q.empty():

            now_node = q.get_nowait()

            for child in now_node:
                if isinstance(child, nltk.tree.Tree):
                    if child.label() == "VP":
                        return child
                    else:
                        q.put(child)

    def find_realtionship(self,obj,node=None,):
        final_result = []
        final_result.append(self.get_trans(obj))
        if node == None:
            node = self.tree.copy(deep= True)
        result = nltk.Tree.fromstring("(root s)")

        VP = self.firstVP(node)
        while result:
            rel_now=[]
            zh_now = []
            if VP is None:
                break
            VP = self.DFSFoundNotVisitedNode(VP)

            # VP_str = " ".join(VP.leaves())
            # find relationship before the first NP
            ls,result = self.DFSFirstNP(node=VP)

            # if self.rel:
            #     if ls == self.rel[-1]:
            #         break

            if ls:
                if len(self.rel)>=1 and ls[0] == self.rel[-1][0] and len(ls) == len(self.rel[0]):
                    continue
                # print(ls)
                rel = " ".join(ls).lower()
                zh_rel = self.get_trans(nltk.Tree.fromstring("(rel " + rel + ")"))
                rel_now.append(rel)
                zh_now.append(zh_rel)
                final_result.append(zh_now)
            if result:
                res = " ".join(result.leaves()).lower()
                rel_now.append(res)
                zh_result = self.get_trans(result)
                # print(zh_result)
                final_result[-1].append(zh_result)
                # 目前，有名词的关系被放入rel中
                if len(self.rel) == 0:
                    self.rel.append(rel_now)
                elif self.rel[-1][0] not in rel_now[0]:
                    self.rel.append(rel_now)
        return final_result
        # NP = self.BFSFirstNP(VP)
        # NP_str = " ".join(NP.leaves())
        # print(VP_str)
        # print(NP_str)
        # print(VP_str.replace(NP_str, ""))

    def DFSFoundNotVisitedNode(self,node):
        result = None
        if node.label() != "X":
            result = node
        for child in node:
            if result is not None:
                break
            if isinstance(child,nltk.tree.Tree):
                result = self.DFSFoundNotVisitedNode(child)
        return result

    def DFSFirstNP(self,ls=None, node =None, result = None):
        '''
        主要用于发现关系
        :param ls:
        :param node:
        :param result:
        :return:
        '''
        # X 表示已经找过了
        if ls == None:
            ls = []
        if node == None:
            node = self.tree
        # 存在result或者已经找过该节点，直接返回
        if result or node.label() == "X":
            return ls,result

        # 处理修饰整句的PP
        has_VB = False
        for child in node:
            if isinstance(child, nltk.tree.Tree):
                if child.label() == ",":
                    continue
                if child.label() == "VB":
                    has_VB = True
                # 处理修饰整句话的PP
                if child.label() == "PP":
                    if has_VB:
                        # 设置子节点，保证不会再次访问
                        for i in child.subtrees():
                            i.set_label("X")
                            # result 可能为空
                        if result:
                            result = nltk.Tree.fromstring("(A "+str(result)+str(child)+")")
                        else:
                            result = child
                if result:
                    break

                if child.label() == 'S' and len(self.rel)>=1:
                    # print(self.rel[-1][0])
                    # print(" ".join(child.leaves()).lower())
                    if self.rel[-1][0] not in " ".join(child.leaves()).lower():
                        pre_rel = self.rel[-1][0]
                        pre_obj = self.rel[-1][1]
                        ls.append(pre_rel)
                        ls.append(pre_obj)

                if child.label() == "NP":
                    # result 不为空，之后不会再进入循环
                    result = child
                    # NP内全部标记为已访问
                    for i in child.subtrees():
                        i.set_label("X")

                ls,result=self.DFSFirstNP(ls, child, result)
                # child内节点全部遍历完毕，没有发现NP
                if result is None:
                    child.set_label("X")


            else:
                if node.label() == 'TO' and self.rel and ("to" not in self.rel[-1][0]):
                    self.rel[-1][1] += " "+child
                ls.append(child)

        return ls,result

    def DFSFirstVP(self,ls=None, node =None, result = None):
        '''
        :param ls:存放结果之前的内容
        :param node:需要寻找的节点
        :param result:最终结果
        :return:ls， result
        '''
        # X 表示已经找过了
        if ls is None:
            ls = []
        if node is None:
            node = self.tree
        # 存在result或者已经找过该节点，直接返回
        if result or node.label() == "X":
            return ls,result
        node.set_label("X")
        for child in node:
            if isinstance(child, nltk.tree.Tree):

                if result:
                    break
                if child.label() == "VP":
                    # result 不为空，之后不会再进入循环
                    result = child

                ls,result=self.DFSFirstNP(ls, child, result)
            else:
                ls.append(child)

        return ls, result

    def drawTree(self):
        self.tree.draw()

    def get_trans(self, node):
        result2 = self.trans.translate(" ".join(node.leaves()), dest="zh-cn", src='en')
        return result2.text

    @staticmethod
    def show(node):
        trans = googletrans.Translator()
        result2 = trans.translate(" ".join(node.leaves()), dest="zh-cn", src='en')
        # result2 = trans.translate("post ", dest="zh-cn", src='en')
        print(result2.text)




# node = None
# UBRINlpExtractor.show(node)
test_str="为了加强城乡规划管理，协调城乡空间布局，改善人居环境，促进城乡经济社会全面协调可持续发展，制定本法。"
# test_str = "城镇供热系统的运行维护管理应制定相应的管理制度、岗位责任制、安全操作规程、设施和设备维护保养手册及事故应急预案，并应定期进行修订。"
test = UBRINlpExtractor(test_str)
#
# firstMinNP_t = test.first_minNP()
firstNP_t = test.BFSFirstNP()
# print(" ".join(firstNP_t.leaves()))
# find_entity_t = test.find_entity()
# find_VP_t = test.firstVP()
# test.drawTree()
test.show(firstNP_t)
# test.show(find_entity_t)
# test.show(find_VP_t)
# # test.show(find_entity_t)
# test.show(firstMinNP_t)
result = test.find_realtionship(firstNP_t)
print(result)
test.drawTree()
#
#
# print(test.rel)
# test.show(test.find_realtionship())

# 对比实验
chi_parser = StanfordParser(path_to_jar='../stanford-parser-full-2018-02-27/stanford-parser.jar',
                            path_to_models_jar='../stanford-parser-full-2018-02-27/stanford-parser-3.9.1-models.jar',
                            model_path='../stanford-parser-full-2018-02-27/stanford-parser-3.9.1-models/edu/stanford/nlp/models/lexparser/chinesePCFG.ser.gz')
data_dir='../stanford-segmenter-2018-02-27/'
segmenter = StanfordSegmenter(path_to_jar=data_dir+"stanford-segmenter-3.9.1.jar",
                              path_to_sihan_corpora_dict=data_dir+"/data", path_to_model=data_dir+"/data/pku.gz",
                              path_to_dict=data_dir+"/data/dict-chris6.ser.gz",
                              java_class='edu.stanford.nlp.ie.crf.CRFClassifier',
                              )
result=segmenter.segment(test_str)
result_ls = result.split()
ch_tree = list(chi_parser.parse(result_ls))[0]
ch_tree.draw()
# print(result)
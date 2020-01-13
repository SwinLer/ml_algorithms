'''根据样本集和特征列表选取最优分切点
生成决策树字典形式和图形
'''

import matplotlib.pyplot as plt


def print_dict(src_dict, level, src_dict_namestr=''):
    if isinstance(src_dict, dict):
        tab_str = '\t'
        for i in range(level):
            tab_str += '\t'
        if 0 == level:
            print(src_dict_namestr, '={')
        for key, value in src_dict.items():
            if isinstance(value, dict):
                has_dict = False
                for k, v in value.items():
                    if isinstance(v, dict):
                        has_dict = True
                if has_dict:
                    print(tab_str, key, ":{")
                    print_dict(value, level + 1)
                else:
                    print(tab_str, key, ':', value)
            else:
                print(tab_str, key, ': ', value, )
        print(tab_str, '}')


'''
样本集剪枝
样本特征列表
feature_type_list=['youth','work','hourse','credit']

样本集例：
samples_list = [ ['youth', 'work_no', 'house_no', '1', 'refuse']
                 ['youth', 'work_no', 'house_no', '2', 'refuse']
                 ['youth', 'work_yes', 'house_no', '2', 'agree']
                 ['youth', 'work_yes', 'house_yes', '1', 'agree']
                 ['youth', 'work_no', 'house_no', '1', 'refuse']
                 ['mid', 'work_no', 'house_no', '1', 'refuse']
                 ['mid', 'work_no', 'house_no', '2', 'refuse']
                 ['mid', 'work_yes', 'house_yes', '2', 'agree']
                 ['mid', 'work_no', 'house_yes', '3', 'agree']
                 ['mid', 'work_no', 'house_yes', '3', 'agree']
                 ['elder', 'work_no', 'house_yes', '3', 'agree']
                 ['elder', 'work_no', 'house_yes', '2', 'agree']
                 ['elder', 'work_yes', 'house_no', '2', 'agree']
                 ['elder', 'work_yes', 'house_no', '3', 'agree']
                 ['elder', 'work_no', 'house_no', '1', 'refuse'] ]

'''


class CTailorSamples:
    def __init__(self, data_list, feat_type_list, feat_type_index, feat_value):
        '''
        :param data_list:样本集 
        :param feat_type_list:样本特征表
        :param feat_type_index:
        :param feat_value:
        '''
        self.data_list = data_list
        self.feat_type_list = feat_type_list
        self.feat_type_index_tailed = feat_type_index
        self.feat_value_tailed = feat_value

        self.tailer_work()

    def get_samples(self):
        '''
        返回样本集 特征表
        '''
        return self.data_list, self.feat_type_list

    def get_all_indexs(self, src_list, dst_value):
        '''
        返回给定元素下标
        :param src_list:元素表
        :param dst_value:选定元素
        '''
        dst_value_index = [i for i, x in enumerate(src_list) if x == dst_value]
        return dst_value_index

    def tailer_work(self):
        '''
        剪裁生成新特征列表
        '''
        del self.feat_type_list[self.feat_type_index_tailed]

        '''裁剪数据'''
        # 获取被删除特征列
        colum_to_del = self.feat_type_index_tailed
        self.feat_value_list = [example[colum_to_del] for example 
                                in self.data_list]

        # 获取行下标
        # 从大行下标开始删除
        rows_to_del = self.get_all_indexs(self.feat_value_list, self.feat_value_tailed)
        rows_to_del.reverse()
        for row in rows_to_del:
            del self.data_list[row]

        # 删除特征列
        for row in range(len(self.data_list)):
            del self.data_list[row][colum_to_del]

        return self.data_list, self.feat_type_list


'''
寻找切分点划分样本
'''


class CCartTree:
    def __init__(self, samples, feat_list, div_label, max_n_feats):
        self.samples = samples
        self.feat_list = feat_list
        self.div_label = div_label
        self.max_n_feats = max_n_feats
        self.tree_dict = {}
        self.create_tree()

    def get_tree_dict(self):
        return self.tree_dict

    def work(self, samples, feat_list, div_label, max_n_feats):
        '''
        给定样本数据集+特征列表，找出最优特征，最优切分点，最优叶节点，次优切分点
        :param samples:样本集
        :param：特征列表
        :return 样本集的最优特征，最优切分点，最优叶节点
        '''
        stat, n_samples = {}, len(samples)
        class_vals = [e[-1] for e in samples]
        class_set = set(class_vals)
        for i in range(len(feat_list)):
            f, stat[f] = feat_list[i], {}  # feature
            for e in samples:
                # feature's value, feature value's class label
                v, c = e[i], e[-1]  
                if v not in stat[f].keys():
                    stat[f][v], stat[f][v]['n'], stat[f][v]['p'] = {}, 0, 0.0
                    stat[f][v][c], stat[f][v][c]['n'], stat[f][v][c]['p'] \
                        = {}, 0, 0.0
                elif c not in stat[f][v].keys():
                    stat[f][v][c], stat[f][v][c]['n'], stat[f][v][c]['p'] \
                        = {}, 0, 0.0
                stat[f][v]['n'] += 1
                stat[f][v]['p'] = stat[f][v]['n']/n_samples
                stat[f][v][c]['n'] += 1
                # update stat[f][v][every c]['p']
                for x in class_set:
                    if x not in stat[f][v].keys():
                        stat[f][v][x], stat[f][v][x]['n'], stat[f][v][x]['p'] = {}, 0, 0
                    stat[f][v][x]['p'] = stat[f][v][x]['n'] / stat[f][v]['n']
                    p = float(stat[f][v][x]['p'])
                    stat[f][v][x]['gini'] = 2*p*(1 - p)
                # update stat[f][v]['gini']
                d1_p, d2_p = stat[f][v]['p'], 1-stat[f][v]['p']
                prob = (class_vals.count(div_label) - stat[f][v]
                        [div_label]['n']) / (n_samples - stat[f][v]['n'])
                d1_gini, d2_gini = stat[f][v][div_label]['gini'], \
                    2*prob*(1 - prob)
                stat[f][v]['gini'] = d1_p * d1_gini + d2_p * d2_gini

            # 选最优特征，最优切分点，最优叶子节点
        min_v_gini, bf_bv = 9257, []
        for i in range(len(feat_list)):
            f = feat_list[i]  # visit every feature
            for v in set([e[i] for e in samples]):
                if min_v_gini > stat[f][v]['gini']:
                    min_v_gini, bf_bv = stat[f][v]['gini'], [(f, v)]
                elif min_v_gini == stat[f][v]['gini']:
                    bf_bv.append((f, v))
        min_c_gini, bf, bv, bc = 9527, None, None, None
        for (f, v) in bf_bv:   # gini相等时选择最优特征
            for c in class_set:
                if min_c_gini > stat[f][v][c]['gini']:
                    min_c_gini = stat[f][v]['gini']
                    bf, bv, bc = f, v, c
                elif min_c_gini == stat[f][v][c]['gini']:
                    if stat[f][v][c]['p'] > stat[bf][bv][bc]['p']:
                        bf, bv, bc = f, v, c
                        
        # 找最优特征的次优分切点
        min_c_gini, better_v = 9527, None
        bf_v_set = set([e[feat_list.index(bf)] for e in samples])
        bf_v_set.remove(bv)
        for v in bf_v_set:   # 多gini相等时 
            if min_c_gini > stat[bf][v]['gini']:
                min_c_gini, better_v = stat[bf][v]['gini'],  v

        # 找最优特征大次优分切点的最优节点
        min_c_gini, better_c = 9527, None
        for c in class_set:
            if min_c_gini > stat[bf][better_v][c]['gini']:
                min_c_gini, better_c = stat[bf][better_v][c]['gini'],  c
            elif min_c_gini == stat[bf][better_v][c]['gini']:
                if stat[bf][better_v][c]['p'] > \
                        stat[bf][better_v][better_c]['p']:
                    better_c = c 
        return bf, bv, bc, better_v, better_c, stat

    def create_tree(self):
        if len(self.feat_list) < self.max_n_feats:
            return None

        # get current tree
        bf, bv, bc, better_v, better_c, stat = self.work(
                                               self.samples,
                                               self.feat_list,
                                               self.div_label,
                                               self.max_n_feats)
        root, rcond, rnode, lcond, lnode = bf, bv, bc, better_v, better_c
        print("better_c:", better_c)

        # get child tree, first to tailor samles
        tailor = CTailorSamples(self.samples,
                                self.feat_list,
                                self.feat_list.index(bf),
                                bv)
        new_samples, new_feat_list = tailor.get_samples()

        cart = CCartTree(new_samples, new_feat_list, self.div_label, self.max_n_feats)
        child_node = cart.get_tree_dict()
        print('child_node', child_node)
        # update current tree left-child-tree
        if child_node is not None and child_node != {}:
            lnode = child_node

        # current tree dict
        self.tree_dict = {}
        self.tree_dict[root] = {}
        self.tree_dict[root][rcond] = rnode
        self.tree_dict[root][lcond] = lnode
        return self.get_tree_dict()

# 节点形状


decisionNode = dict(boxStyle="round4", color='r', fc='0.9')
leafNode = dict(boxstyle="circle", color='m')
arrow_args = dict(arrowstyle="<-", color='g')


def plot_node(node_txt, center_point, parent_point, node_style):
    '''内部函数
    绘制父子节点及箭头和文本
    '''
    createPlot.ax1.annotate(node_txt, xy=parent_point,
                           xycoords='axes fraction',
                           xytext=center_point, textcoords='axes fraction',
                           va="center", ha="center", bbox=node_style,
                           arrowprops=arrow_args)


def get_leafs_num(tree_dict):
    '''
    内部函数
    获取叶节点个数
    '''
    leafs_num = 0
    if len(tree_dict.keys()) == 0:
        print("input tree dict is void!")
        return 0
    
    root = list(tree_dict.keys())[0]
    # 键值即该节点所有子树
    child_tree_dict = tree_dict[root]
    for key in child_tree_dict.keys():
        if type(child_tree_dict[key]).__name__ == 'dict':
            leafs_num += get_leafs_num(child_tree_dict[key])
        else:
            leafs_num += 1
    return leafs_num


def get_tree_max_depth(tree_dict):
    '''
    内部函数
    返回树层数
    '''
    max_depth = 0
    if len(tree_dict.keys()) == 0:
        print('input tree_dict is void!')
        return 0

    root = list(tree_dict.keys())[0]

    child_tree_dict = tree_dict[root]
    for key in child_tree_dict.keys():
        this_path_depth = 0
        if type(child_tree_dict[key]).__name__ == 'dict':
            # 子树为字典型，则非叶节点，当前分支层数加上子树最深层数
            this_path_depth = 1 + get_tree_max_depth(child_tree_dict[key])
        else:
            # 子树非字典型，为叶节点
            this_path_depth = 1
        if this_path_depth > max_depth:
            max_depth = this_path_depth

    return max_depth


def plot_mid_text(center_point, parent_point, txt_str):
    '''
    内部函数
    计算父子节点中间位置
    '''
    x_mid = (parent_point[0] - center_point[0]) / 2.0 + center_point[0]
    y_mid = (parent_point[1] - center_point[1]) / 2.0 + center_point[1]
    createPlot.ax1.text(x_mid, y_mid, txt_str)
    return


def plotTree(tree_dict, parent_point, node_txt):
    '''
    内部函数
    绘制树
    '''
    leafs_num = get_leafs_num(tree_dict)
    root = list(tree_dict.keys())[0]
    center_point = (plotTree.xOff + (1.0 + float(leafs_num))/2.0/plotTree.totalW, plotTree.yOff)
    plot_mid_text(center_point, parent_point, node_txt)
    plot_node(root, center_point, parent_point, decisionNode)
    child_tree_dict = tree_dict[root]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    
    for key in child_tree_dict.keys():
        if type(child_tree_dict[key]).__name__ == 'dict':
            plotTree(child_tree_dict[key], center_point, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plot_node(child_tree_dict[key], (plotTree.xOff, plotTree.yOff),
                        center_point, leafNode)
            plot_mid_text((plotTree.xOff, plotTree.yOff), center_point, str(key))
    
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD
    return


def createPlot(tree_dict):
    '''
    绘制决策树
    '''
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(get_leafs_num(tree_dict))
    plotTree.totalD = float(get_tree_max_depth(tree_dict))
    if plotTree.totalW == 0:
        print('tree_dict is void!')
        return
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(tree_dict, (0.5, 1.0),  '')
    plt.show()


def create_samples(): 
    '''
    训练样本集
    每个样本由n个特征值和n+1个分类标签组成
    分类标签取值为'refuse'和'agree'
    '''
    data_list = [['youth', 'no',  'no',   '1', 'refuse'],  
                 ['youth', 'no',  'no',   '2', 'refuse'],  
                 ['youth', 'yes', 'no',   '2', 'agree'],  
                 ['youth', 'yes', 'yes',  '1', 'agree'],  
                 ['youth', 'no',  'no',   '1', 'refuse'],  
                 ['mid',   'no',  'no',   '1', 'refuse'],  
                 ['mid',   'no',  'no',   '2', 'refuse'],  
                 ['mid',   'yes', 'yes',  '2', 'agree'],  
                 ['mid',   'no',  'yes',  '3', 'agree'],  
                 ['mid',   'no',  'yes',  '3', 'agree'],  
                 ['elder', 'no',  'yes',  '3', 'agree'],  
                 ['elder', 'no',  'yes',  '2', 'agree'],  
                 ['elder', 'yes', 'no',   '2', 'agree'],  
                 ['elder', 'yes', 'no',   '3', 'agree'],  
                 ['elder', 'no',  'no',   '1', 'refuse']]
    feat_list = ['age', 'working', 'house', 'credit']  
    return data_list, feat_list


if __name__ == '__main__':
    data_list, feat_list = create_samples()

    cart = CCartTree(data_list, feat_list, 'agree', 3)
    tree_dict = cart.get_tree_dict()
    print_dict(tree_dict, 0, 'tree_dict')

    createPlot(tree_dict)

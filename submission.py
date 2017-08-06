import pickle
import re

# from sklearn import linear_model
from sklearn import tree
# import sklearn.naive_bayes
# from sklearn.linear_model import LogisticRegression
# from sklearn import neighbors

# 音标对应表，把所有音标map到一个int上面去，总共有39个音标，[0,14]是元音，[15,38]是辅音
PHONEMES = {'AA': 0, 'AE': 1, 'AH': 2, 'AO': 3, 'AW': 4, 'AY': 5, 'EH': 6, 'ER': 7,
'EY': 8, 'IH': 9, 'IY': 10, 'OW': 11, 'OY': 12, 'UH': 13, 'UW': 14, 'P': 15,
'B': 16, 'CH': 17, 'D': 18, 'DH': 19, 'F': 20, 'G': 21, 'HH': 22, 'JH': 23, 'K': 24,
'L': 25, 'M': 26, 'N': 27, 'NG': 28, 'R': 29, 'S': 30, 'SH': 31, 'T': 32, 'TH': 33,
'V': 34, 'W': 35, 'Y': 36, 'Z': 37, 'ZH': 38}

# 前缀表，元组
PRE = ('zo', 'with', 'vor', 'volv', 'volt', 'volcan', 'vol', 'voke', 'voc', 'vivi', 'viv',
'vita', 'vis', 'vinc', 'vid', 'vict', 'vicis', 'vice', 'vic', 'vi', 'verv', 'vert', 'vers',
'veri', 'verb', 'ver', 'vent', 'ven', 'veh', 'vect', 'valu', 'vali', 'vale', 'vade', 'vac',
'up', 'uni', 'under', 'un', 'umbraticum', 'umber', 'ultima', 'typ', 'ty', 'twi', 'turbo',
'tribute', 'trib', 'tri', 'treat', 'trans，ad', 'trans', 'trai', 'tract', 'tra', 'tox', 'tort',
'tors', 'tor', 'tom', 'tire', 'ting', 'tin', 'tig', 'thir', 'thet', 'thesis', 'therm', 'theo',
'the', 'test', 'terra', 'terr', 'term', 'tera', 'tent', 'tens', 'tend', 'ten', 'tempo', 'tem',
'tele', 'teg', 'tect', 'tang', 'tain', 'tag', 'tact', 'syn', 'sym', 'syl', 'sus', 'sur', 'supra',
'super', 'sup', 'sump', 'sume', 'sum', 'sug', 'suf', 'sue', 'suc', 'sub', 'stry', 'struct', 'stru',
'stroy', 'string', 'strict', 'strain', 'stit', 'stige', 'sti', 'stead', 'stat', 'stant', 'stand',
'stan', 'stab', 'sta', 'st', 'spir', 'spic', 'spi', 'sphere', 'sper', 'spect', 'spec', 'soph',
'somn', 'solv', 'solut', 'solus', 'solu', 'sol', 'soci', 'sist', 'simul', 'simil', 'signi', 'sign',
'sid', 'sex', 'sess', 'ses', 'serv', 'sequ', 'sept', 'sent', 'sens', 'sen', 'semi', 'sed', 'secu',
'sect', 'secr', 'sec', 'se', 'script', 'scrib', 'scope', 'scio', 'scientia', 'sci', 'scen', 'satis',
'sat', 'sanct', 'sanc', 'salv', 'salu', 'sacr', 'rupt', 'roga', 'rog', 'risi', 'ridi', 'ri', 'retro',
'reg', 'recti', 're', 'quis', 'quir', 'quip', 'quint', 'quest', 'quer', 'quat', 'quad', 'pute', 'pur',
'punct', 'puls', 'psych', 'pseudo', 'proto', 'pro', 'prin', 'prime', 'prim', 'pri', 'prehendere',
'pre', 'pound', 'pot', 'post', 'pos', 'portion', 'port', 'pop', 'pond', 'pon', 'poly', 'poli', 'pod',
'pneumon', 'pneuma', 'ply', 'plus', 'plur', 'plu', 'plore', 'pli', 'plais', 'plac', 'pict', 'pico',
'photo', 'phot', 'phon', 'phobos', 'phobia', 'phlegma', 'phil', 'phen', 'phe', 'phas', 'phant',
'phan', 'phage', 'peri', 'per', 'penta', 'pens', 'pene', 'pend', 'pel', 'pedo', 'ped', 'patr',
'pathy', 'path', 'pater', 'pat', 'pass', 'pare', 'para', 'pan', 'paleo', 'pair', 'pac', 'over',
'out', 'ortho', 'oper', 'op', 'onym', 'omni', 'ology', 'oligo', 'of', 'oct', 'oc', 'ob', 'nym',
'numisma', 'numer', 'nox', 'nov', 'non', 'nomin', 'nomen', 'nom', 'noc', 'neur', 'neo', 'neg',
'ne', 'nat', 'nasc', 'nano', 'nai', 'n', 'myria', 'multi', 'mov', 'mot', 'mort', 'morph', 'mor',
'mono', 'mon', 'mob', 'mit', 'miss', 'mis', 'min', 'milli', 'mill', 'migra', 'mid', 'micro', 'metr',
'meter', 'meta', 'meso', 'mer', 'ment', 'mem', 'mega', 'medi', 'med', 'matri', 'mari', 'mar', 'manu',
'mania', 'mand', 'man', 'male', 'mal', 'main', 'magni', 'magn', 'macro', 'macr', 'macer', 'lut',
'lust', 'lus', 'lun', 'lum', 'lude', 'luc', 'lot', 'loqu', 'logo', 'log', 'locut', 'loco', 'loc',
'liver', 'liter', 'lig', 'lide', 'liber', 'lex', 'levi', 'leg', 'lect', 'leag', 'lav', 'lau', 'labor',
'kilo', 'juven', 'just', 'junct', 'jug', 'judice', 'join', 'ject', 'jac', 'ir', 'intro', 'intra',
'inter', 'intel', 'infra', 'in', 'im', 'il', 'ignis', 'ig', 'ics', 'hypn', 'hyper', 'hydro', 'hydra',
'hydr', 'human', 'hum', 'homo', 'hex', 'hetero', 'hes', 'here', 'her', 'hemo', 'hemi', 'hema', 'helio',
'hecto', 'hect', 'heal', 'hale', 'gress', 'greg', 'gree', 'grav', 'grat', 'graph', 'gram', 'graf',
'grad', 'gor', 'gnos', 'gnant', 'glu', 'glot', 'gloss', 'glo', 'gin', 'giga', 'gest', 'germ', 'geo',
'gen', 'ge', 'gastro', 'gastr', 'gam', 'fuse', 'fuge', 'frai', 'frag', 'fract', 'fort', 'form',
'fore', 'forc', 'for', 'flux', 'fluv', 'fluc', 'flu', 'flict', 'flex', 'flect', 'fix', 'fit', 'fin',
'fili', 'fila', 'fig', 'fide', 'fid', 'fic', 'fess', 'fer', 'femto', 'feign', 'feder', 'fect', 'fec',
'feat', 'fea', 'fas', 'fant', 'fan', 'fals', 'fall', 'fain', 'fact', 'fac', 'fa', 'extro', 'extra',
'exter', 'ex', 'ev', 'eu', 'et', 'es', 'erg', 'equi', 'epi', 'enni', 'end', 'en', 'em', 'ecto', 'eco',
'ec', 'dys', 'dynam', 'dy', 'dura', 'duct', 'duc', 'dox', 'dorm', 'dont', 'don', 'domin', 'doct',
'doc', 'div', 'dit', 'dis', 'dign', 'dif', 'dict', 'dic', 'dia', 'di', 'deun', 'derm', 'dent', 'demo',
'demi', 'dem', 'dei', 'deco', 'deci', 'deca', 'dec', 'de', 'cyclo', 'cycl', 'cuse', 'cus', 'curs',
'curr', 'cura', 'cur', 'cru', 'crit', 'cret', 'cresc', 'cred', 'crease', 'crea', 'cre', 'crat',
'cracy', 'cour', 'counter', 'cosm', 'cort', 'corp', 'cord', 'cor', 'cop', 'contro', 'contre',
'contra', 'contr', 'con', 'com', 'coll', 'col', 'cogn', 'cog', 'co', 'clusclaus', 'clud', 'clin',
'clam', 'claim', 'civ', 'cit', 'cise', 'cis', 'circum', 'circu', 'cip', 'cide', 'cid', 'chron',
'chrom', 'cess', 'cept', 'centri', 'centr', 'centi', 'cent', 'ceiv', 'ceed', 'cede', 'ced', 'ceas',
'caut', 'cause', 'caus', 'cath', 'cata', 'cat', 'cas', 'carn', 'cardi', 'capt', 'capit', 'cap',
'calor', 'cad', 'by', 'brev', 'bio', 'bine', 'bin', 'biblio', 'bibli', 'bibl', 'bi', 'bene', 'belli',
'be', 'bar', 'auto', 'aut', 'aus', 'aur', 'aug', 'audi', 'aud', 'auc', 'at', 'astr', 'aster', 'as',
'arch', 'ar', 'aqu', 'apo', 'aph', 'ap', 'antico', 'anti', 'anthrop', 'ante', 'ant', 'ano', 'annu',
'ann', 'anim', 'ang', 'andro', 'andr', 'ana', 'an', 'amor', 'ami', 'ambul', 'ambi', 'am', 'alter',
'alt', 'allo', 'ali', 'albo', 'alb', 'al', 'agro', 'agri', 'agi', 'ag', 'af', 'aero', 'aer', 'ad',
'acu', 'act', 'acri', 'acid', 'acer', 'ac', 'abs', 'ab')

def get_selected_classifier():
    """
    抽象出一个接口给submitted train和 practice train调用，如果想要选择不同的分类器，只需要改这里
    即可。改完后，无需再在submitted train和practice train两者之间协调。

    Returns:
        clf (classifier): 选择的分类器
    """
    # 这两个需要特征为非负数，基本可以忽略
    # clf = linear_model.BayesianRidge()
    # clf = sklearn.naive_bayes.MultinomialNB()

    # clf = tree.DecisionTreeClassifier(criterion='gini')
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    # clf = sklearn.naive_bayes.GaussianNB()
    # clf = sklearn.naive_bayes.BernoulliNB()
    # clf = LogisticRegression()
    # clf = neighbors.KNeighborsClassifier(algorithm='kd_tree')
    
    return clf

def word_pre_index(word):
    """
    查看前缀表，看看word是否有相关的前缀

    Args:
        word (str): The word

    Returns:
        index (int): if found in PRE, return the index of the PRE. if not, return -1
    """
    for i in PRE:
        if word.startswith(i.upper()):
            return PRE.index(i)
    return -1

def c_v_comb_hashing(tu):
    """
    Generate a hash number from the consonant and vowel combination in one tuple.
    Rule: 每个音节代表的int占hashnu的两位，用十进制表示，体现出顺序关系

    Args:
        tu (tuple): one item in the con_vol_combination list.
    Returns:
        hashnu (int): the hash number of the tuple.


    example:
        step1. 原始数据：LEARNING:L ER N IH NG
        step2. 音标list：[L, ER, N, IH, NG]
        step3. 音标map：[25, 7, 27, 9, 28] （查submission.py里面的音标表）
        step4. 辅音+元音组合系列：[(25,7), (27,9)]. （这里没有考虑28，因为考虑的组合是以元音作为结尾）
        step5. 哈希一下：[2507, 2709]
        step6. 最后的feature matrix的后四位：[2507, 2709, -1,-1]
    """
    hashnu = 0
    l = len(tu)
    for i in range(l):
        hashnu = hashnu + tu[i] * (10 ** (2 * (l - i - 1)))

    return hashnu

def getInfoOfPronsFromTrain(word,prons):
    """
    Calculate the features and label of one training data sample

    Args:
        word (str): The word spelling of all the upper case.
        prons (str): A str consisting the prons of the current word.

        eg. 'AVALOS:AA0 V AA1 L OW0 Z'

    Returns:
        features (list): The feature list of the training data sample.
            相关的feature构造就在这个函数里面去实现就好
            features 对应的意义：
            features = vowels_seq + combhash_seq + [vowels_count, pre_index] 

            [0,3] 该单词从左到右的元音序列，题目已经限制元音数小于5，不存在赋值为-1 
                eg. 若发音从左到右的元音是是AE,AH， 则结果为 [1,3,-1,-1]
            [4,8] 从左到右辅音+元音组合，因为元音数小于5，所以组合数也一定小于5。用c_v_comb_hashing方法来构建有意义的部分，
            不存在赋值为-1
            9. 该单词中元音的总数(int)
            10. 该单词若有前缀，在前缀表的index，若没有在给出前缀中，则赋值为-1
        label (int): The class label of the training data sample. 
            也就是重音位置在元音的index，从1开始.Range:{0,1,2,3,4}, 0表示没有重音(虽然好像在训练集中不存在)
    """

    # 开始构造features

    # 该单词的每个元音元音以及其前面辅音的组合
    # 用了嵌套+正则，想看懂有点僵硬
    # 记录下思路，防止以后忘记
    # 1. 用正则把数字作为边界分开
    # 2. 把分开后的最后一个舍弃（因为最后一个不再含有元音）
    # 3. 字符串处理把每一项的首尾空格去掉，再以空格为边界split
    # 4. 通过字典映射把字符串转成对应的int
    # 5. 把item转成tuple
    # 实在不懂可以忽略过程。
    # 有空的话也可以验证一下这一步是否正确
    con_vol_combination = [tuple(PHONEMES[pron] for pron in x.strip().split(' ')) for x in re.split(r'\d{0,2}', prons)[:-1]]
    # 因为con_vol_combination这个list，每一项是一个'辅音+元音'的组合，每一项的最后一个都是元音，并且按照顺序排列
    # 所以接下来的很多feature都可以通过这个list来获得，避免了开销较大的迭代
    # 元音个数

    vowels_count = len(con_vol_combination)
    # 元音出现的序列
    vowels_seq = [x[-1] for x in con_vol_combination]
    # 辅音+元音组合的序列
    combhash_seq = [c_v_comb_hashing(tu) for tu in con_vol_combination]

    # 两个seq补充至4位
    while len(vowels_seq) < 4:
        assert len(vowels_seq) == len(combhash_seq)
        vowels_seq.append(-1)
        combhash_seq.append(-1)

    pre_index = word_pre_index(word)

    # 在这里构造出features，想要改的话也就在这里进行增删
    features = vowels_seq + combhash_seq + [vowels_count, pre_index] 

    # 获得重音位置，get到label
    label = 0
    for p in prons.split(' '):
        if p[-1].isdigit():
            label = label + 1
            if p[-1] == '1':
                break
    else:
        # no primary stress
        label = 0

    return features, label

def training_preprocess(data):
    """
    preprocess the raw training data from the helper.read_data() method.
    生成训练所需要的特征矩阵x_train和训练集对应的label矩阵y_train
    
    Args:
        data (list): The list data read by helper.read_data() method.

    Returns:
        x_train (array): of size [n_samples, n_features]. eg.[[1,2,3], [4,5,6]]
        y_train (array): of size [n_samples]. eg. [1, 2]
                        holding the class labels for the training samples
    """

    """
    y_train item ranges: {0,1,2,3,4}. 因为要求中只考虑元音数量少于5个的情况，然后在元音里面的
    index也是从1开始算起。0表示没有重音
    """

    x_train = []
    y_train = []

    for d in data:
        word, prons = d.split(':')
        features, label = getInfoOfPronsFromTrain(word,prons)
        x_train.append(features)
        y_train.append(label)

    return x_train, y_train

def getInfoFromTest(word, prons):
    """
    eg. LEARNING:L ER N IH NG
    """
    pre_index = word_pre_index(word)
    mapprons = [PHONEMES[p] for p in prons.split(' ')]

    vowels_count = sum([x < 15 for x in mapprons])

    begin,end = 0, 0
    con_vol_combination = []
    count = vowels_count
    while count > 0:
        if mapprons[end] < 15:
            con_vol_combination.append(tuple(mapprons[begin:end+1]))
            count = count - 1
            end = end + 1
            begin = end
        else:
            end = end + 1
    
    vowels_seq = [x[-1] for x in con_vol_combination]
    combhash_seq = [c_v_comb_hashing(tu) for tu in con_vol_combination]

    while len(vowels_seq) < 4:
        assert len(vowels_seq) == len(combhash_seq)
        vowels_seq.append(-1)
        combhash_seq.append(-1)

    return vowels_seq + combhash_seq + [vowels_count, pre_index] 

def testing_preprocess(data):
    """
    Get the features from the raw testing data read through helper.read_data() method

    Args:
        data (list): The list data read by helper.read_data() method.

    Returns:
        x_test (array): of size [n_samples, n_features]. eg.[[1,2,3], [4,5,6]]
    """

    return [getInfoFromTest(x.split(':')[0], x.split(':')[1]) for x in data]


################# Submitted training #################

def train(data, classifier_file):# do not change the heading of the function
    x_train, y_train = training_preprocess(data)
    clf = get_selected_classifier()

    clf.fit(x_train, y_train)
    output = open(classifier_file, 'wb')
    pickle.dump(clf, output)
    output.close()

################# Submitted testing #################

def test(data, classifier_file):# do not change the heading of the function
    pkl_file = open(classifier_file, 'rb')
    dt = pickle.load(pkl_file)
    r = dt.predict(testing_preprocess(data))
    pkl_file.close()
    return list(r)
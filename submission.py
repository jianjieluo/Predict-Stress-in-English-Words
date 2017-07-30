from sklearn import tree
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import f1_score 
import pickle
import re
import numpy as np

# 音标对应表，把所有音标map到一个int上面去，总共有39个音标，[0,14]是元音，[15,38]是辅音
PHONEMES = {'AA': 0, 'AE': 1, 'AH': 2, 'AO': 3, 'AW': 4, 'AY': 5, 'EH': 6, 'ER': 7,
'EY': 8, 'IH': 9, 'IY': 10, 'OW': 11, 'OY': 12, 'UH': 13, 'UW': 14, 'P': 15,
 'B': 16, 'CH': 17, 'D': 18, 'DH': 19, 'F': 20, 'G': 21, 'HH': 22, 'JH': 23, 'K': 24,
  'L': 25, 'M': 26, 'N': 27, 'NG': 28, 'R': 29, 'S': 30, 'SH': 31, 'T': 32, 'TH': 33,
   'V': 34, 'W': 35, 'Y': 36, 'Z': 37, 'ZH': 38}


def s_has_pre(s):
    pre = tuple("an,dis,in,ig,il,im,ir,ne,n,non,neg,un,male,mal,pseudo,mis,de\
    un,anti,ant,contra,contre,contro,counter,ob,oc,of,op,with,by,circum,\
    circu,de,en,ex,ec,es,fore,in,il,im,ir,inter,intel,intro,medi,med,mid,out,\
    over,post,pre,pro,sub,suc,suf,sug,sup,sur,sus,sur,trans,under,up,\
    ante,anti,ex,fore,mid,medi,post,pre,pri,out,over,post,pre,pro,sub,suc,suf,\
    sug,sum,sup,sur,sus,super,sur,trans,under,up,ante,anti,ex,fore,mid,medi,post,\
    pre,pri,pro,re,by,extra,hyper,out,over,sub,suc,sur,super,sur,under,vice,com,\
    cop,con,cor,co,syn,syl,sym,al,over,pan,ex,for,re,se,dia,per,pel,trans，ad,\
    ac,af,ag,an,ap,ar,as,at,ambi,bin,di,twi,tri,thir,deca,deco,dec,deci,hecto,\
    hect,centi,kilo,myria,mega,micro,multi,poly,hemi,demi,semi,pene,arch,auto,bene,\
    eu,male,mal,macro,magni,micro,aud,bio,ge,phon,tele,\
    ac,ad,af,ag,al,an,ap,as,at,an,ab,abs,acer,acid,acri,act,ag,acu,aer,aero,ag,agi,\
    ig,act,agri,agro,alb,albo,ali,allo,alter,alt,am,ami,amor,ambi,ambul,ana,ano,andr,\
    andro,ang,anim,ann,annu,enni,ante,anthrop,anti,ant,anti,antico,apo,ap,aph,aqu,arch,\
    aster,astr,auc,aug,aut,aud,audi,aur,aus,aug,auc,aut,auto,bar,be,belli,bene,bi,bine,\
    bibl,bibli,biblio,bio,bi,brev,cad,cap,cas,ceiv,cept,capt,cid,cip,cad,cas,calor,capit,\
    capt,carn,cat,cata,cath,caus,caut,cause,cuse,cus,ceas,ced,cede,ceed,cess,cent,centr,\
    centri,chrom,chron,cide,cis,cise,circum,cit,civ,clam,claim,clin,clud,clusclaus,co,cog,\
    col,coll,con,com,cor,cogn,gnos,com,con,contr,contra,counter,cord,cor,cardi,corp,cort,\
    cosm,cour,cur,curr,curs,crat,cracy,cre,cresc,cret,crease,crea,cred,cresc,cret,crease,\
    cru,crit,cur,curs,cura,cycl,cyclo,de,dec,deca,dec,dign,dei,div,dem,demo,dent,dont,derm,\
    di,dy,dia,dic,dict,dit,dis,dif,dit,doc,doct,domin,don,dorm,dox,duc,duct,dura,dynam,dys,\
    ec,eco,ecto,en,em,end,epi,equi,erg,ev,et,ex,exter,extra,extro,fa,fess,fac,fact,fec,fect,\
    fic,fas,fea,fall,fals,femto,fer,fic,feign,fain,fit,feat,fid,fid,fide,feder,fig,fila,fili,\
    fin,fix,flex,flect,flict,flu,fluc,fluv,flux,for,fore,forc,fort,form,fract,frag,frai,fuge,\
    fuse,gam,gastr,gastro,gen,gen,geo,germ,gest,giga,gin,gloss,glot,glu,glo,gor,grad,gress\
    ,gree,graph,gram,graf,grat,grav,greg,hale,heal,helio,hema,hemo,her,here,hes,hetero,hex\
    ,ses,sex,homo,hum,human,hydr,hydra,hydro,hyper,hypn,an,ics,ignis,in,im,in,im,il,ir,infra\
    ,inter,intra,intro,ty,jac,ject,join,junct,judice,jug,junct,just,juven,labor,lau,lav,lot\
    ,lut,lect,leg,lig,leg,levi,lex,leag,leg,liber,liver,lide,liter,loc,loco,log,logo,ology\
    ,loqu,locut,luc,lum,lun,lus,lust,lude,macr,macer,magn,main,mal,man,manu,mand,mania,mar\
    ,mari,mer,matri,medi,mega,mem,ment,meso,meta,meter,metr,micro,migra,mill,kilo,milli,min\
    ,mis,mit,miss,mob,mov,mot,mon,mono,mor,mort,morph,multi,nano,nasc,nat,gnant,nai,nat,nasc\
    ,neo,neur,nom,nom,nym,nomen,nomin,non,non,nov,nox,noc,numer,numisma,ob,oc,of,op,oct,oligo\
    ,omni,onym,oper,ortho,over,pac,pair,pare,paleo,pan,para,pat,pass,path,pater,patr,path,pathy\
    ,ped,pod,pedo,pel,puls,pend,pens,pond,per,peri,phage,phan,phas,phen,fan,phant,fant,phe,phil\
    ,phlegma,phobia,phobos,phon,phot,photo,pico,pict,plac,plais,pli,ply,plore,plu,plur,plus,pneuma\
    ,pneumon,pod,poli,poly,pon,pos,pound,pop,port,portion,post,pot,pre,pur,prehendere,prin,prim,\
    prime,pro,proto,psych,punct,pute,quat,quad,quint,penta,quip,quir,quis,quest,quer,re,reg,recti\
    ,retro,ri,ridi,risi,rog,roga,rupt,sacr,sanc,secr,salv,salu,sanct,sat,satis,sci,scio,scientia,\
    scope,scrib,script,se,sect,sec,sed,sess,sid,semi,sen,scen,sent,sens,sept,sequ,secu,sue,serv,\
    sign,signi,simil,simul,sist,sta,stit,soci,sol,solus,solv,solu,solut,somn,soph,spec,spect,spi,\
    spic,sper,sphere,spir,stand,stant,stab,stat,stan,sti,sta,st,stead,strain,strict,string,stige,\
    stru,struct,stroy,stry,sub,suc,suf,sup,sur,sus,sume,sump,super,supra,syn,sym,tact,tang,tag,tig,\
    ting,tain,ten,tent,tin,tect,teg,tele,tem,tempo,ten,tin,tain,tend,tent,tens,tera,term,terr,terra,\
    test,the,theo,therm,thesis,thet,tire,tom,tor,tors,tort,tox,tract,tra,trai,treat,trans,tri,trib,\
    tribute,turbo,typ,ultima,umber,umbraticum,un,uni,vac,vade,vale,vali,valu,veh,vect,ven,vent,ver,\
    veri,verb,verv,vert,vers,vi,vic,vicis,vict,vinc,vid,vis,viv,vita,vivi,voc,voke,vol,volcan,volv\
    ,volt,vol,vor,with,zo".replace(" ","").split(","))
    for i in pre:
        if s.startswith(i.upper()):
            return True
    return False

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
            0. 该单词中元音的总数(int)
            1. 该单词是否具有前缀(bool)
            2. 该单词的元音序列(tuple)，eg. (1,3)表示该单词的有两个元音，从左到右是AE,AH
            3. item是tuple的一个list。item分别顺序表示该单词的每个元音元音以及其前面辅音的组合。
                eg. NONPOISONOUS:N AA0 N P OY1 Z AH0 N AH0 S
                则根据拼音将分成：[N, AA], [N, P, OY], [Z, AH], [N, AH]
                再转过来就变成：[(27, 0), (27, 15, 12), (37, 2), (27, 2)]
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
    # 元音出现的序wels_counto列
    # vowels_seq = tuple([x[-1] for x in con_vol_combination])
    vowels_seq = [x[-1] for x in con_vol_combination]
    while len(vowels_seq) < 4:
        vowels_seq.append(-1)

    is_has_pre = s_has_pre(word)

    # 在这里构造出features，想要改的话也就在这里进行增删
    # features = [vowels_count, is_has_pre, vowels_seq, con_vol_combination]
    features = [vowels_count, is_has_pre] + vowels_seq

    # 获得重音位置，get到label
    label = 0
    index = prons.find('1')
    if index != -1:
        # 这句有一定数据依赖性，因为所有元音都是两个字符
        label = 1 + vowels_seq.index(PHONEMES[prons[index-2:index]])

    return features, label

def training_preprocess(data):
    """
    preprocess the data from the helper.read_data() method.
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
    hasPre = s_has_pre(word)
    mapprons = [PHONEMES[p] for p in prons.split(' ')]
    vowels_count = 0
    vowels_seq = []
    for x in mapprons:
        if x < 15:
            vowels_count = vowels_count + 1
            vowels_seq.append(x)
    
    while len(vowels_seq) < 4:
        vowels_seq.append(-1)

    return [vowels_count, hasPre] + vowels_seq


def testing_preprocess(data):
    """
    Get the features from the testing data read through helper.read_data() method

    Args:
        data (list): The list data read by helper.read_data() method.

    Returns:
        x_test (array): of size [n_samples, n_features]. eg.[[1,2,3], [4,5,6]]
    """

    return [getInfoFromTest(x.split(':')[0], x.split(':')[1]) for x in data]


################# training #################

def train(data, classifier_file):# do not change the heading of the function
    x_train, y_train = training_preprocess(data)
    clf = tree.DecisionTreeClassifier(criterion='gini')

    clf.fit(x_train, y_train)
    output = open(classifier_file, 'wb')
    pickle.dump(clf, output)
    output.close()

################# testing #################

def test(data, classifier_file):# do not change the heading of the function
    pkl_file = open(classifier_file, 'rb')
    dt = pickle.load(pkl_file)
    r = dt.predict(testing_preprocess(data))
    pkl_file.close()
    return list(r)
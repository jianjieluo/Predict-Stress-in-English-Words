import helper
import nltk
import pandas as pd
import submission
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.metrics import f1_score


def get_Vocab(s):
    return s.split(":")
    
def s_has_pre(s):
    pre = "an,dis,in,ig,il,im,ir,ne,n,non,neg,un,male,mal,pseudo,mis,de\
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
    ,volt,vol,vor,with,zo".replace(" ","").split(",")
    #print(pre[0].upper())
    for i in pre:
        if s.startswith(i.upper()):
            return 1
    return 0
def s_has_end(s):
    end = "ee,ese,esque,se,eer,ique,ty,less,ness,ly,ible,able,ion,ic,ical,al,ian,ic,\
    ion,ity,ment,ed,es,er,est,or,ary,ory,ous,cy,ry,ty,al,ure,ute,ble,ar,ly,less,ful,ing,\
    ,inal,tion,sion,osis,oon,sce,\
    que,ette,eer,ee,aire,able,ible,acy,cy,ade,age,al,al,ial,ical,an,ance,ence,ancy,\
    ency,ant,ent,ant,ent,ient,ar,ary,ard,art,ate,ate,ate,ation,cade,drome,ed,ed,en,en,\
    ence,ency,er,ier,er,or,er,or,ery,es,ese,ies,es,ies,ess,est,iest,fold,ful,ful,fy,ia,\
    ian,iatry,ic,ic,ice,ify,ile,ing,ion,ish,ism,ist,ite,ity,ive,ive,ative,itive,ize,less,\
    ly,ment,ness,or,ory,ous,eous,ose,ious,ship,ster,ure,ward,wise,ize,phy,ogy,ity,ion,ic,ical,al".replace(" ","").split(",")
    for i in end:
        if s.endswith(i.upper()):
            return 1
    return 0
def rip_number(s):
    if s[-1].isdigit():
        s = s[:-1]
    return s
def get_Slice(s,r,vowel,cons):
    vocab,phonemes = get_Vocab(s)
    #print(vocab)
    phonemes = phonemes.split(" ")
    stress_matrix = []
    words = vowel+cons
    has_pre = 0
    has_end = 0
    #A SLICE OF FEATURE
    #1.音节数量 2.音节组合 5.是否有前缀 6.后缀是否为这些 7.prime
    current = []
    vowels = [-1,-1,-1,-1]
    vowel_pos = [-1,-1,-1,-1,-1,-1,-1,-1]
#    vowel_pos2 = [-1,-1,-1,-1]
    count_vowel = 0
    for i in range(len(phonemes)):
        if phonemes[i][-1].isdigit():
            stress_matrix.append(phonemes[i][-1])
            vowels[count_vowel]=vowel.index(phonemes[i][:-1])
            if i>0:
                vowel_pos[2*count_vowel] = words.index(rip_number(phonemes[i-1]))
            if i+1<len(phonemes):
                vowel_pos[1+2*count_vowel] = words.index(rip_number(phonemes[i+1]))
            count_vowel+=1
            #if -1 in vowel:
            #    vowel_r[vowel_r.index(-1)] = vowel.index(phonemes[i][:-1])
            #if -1 in vowel_pos:
            #    vowel_pos[vowel_pos.index(-1)] = i
    if "2" in stress_matrix:
        prime_pos = stress_matrix.index("2")    
    elif "1" in stress_matrix:
        prime_pos = stress_matrix.index("1")
#    has_2 =0
#    has_3 =0
#    if len(stress_matrix)==2:
#        has_2 =1
#    if len(stress_matrix)==3:
#        has_3 =1
    has_pre = s_has_pre(vocab)
    has_end = s_has_end(vocab)
#    if len(stress_matrix)==3:
#        print(prime_pos==0)
#    print([len(stress_matrix)]+vowels+[has_pre,has_end,prime_pos])
#    if prime_pos ==0:
#        print(phonemes)
#    r.append([len(stress_matrix),len(phonemes)]+vowels+vowel_pos+vowel_pos2+[vocab,has_pre,has_end,prime_pos+1])
    r.append([len(stress_matrix)]+vowels+vowel_pos+[vocab,has_pre,has_end,prime_pos+1])
def get_Slice2(s,r,vowel,cons):
    vocab,phonemes = get_Vocab(s)
    phonemes = phonemes.split(" ")
    words = vowel+cons
    #A SLICE OF FEATURE, 1.NUMBER_OF_VOW 2.NUMBER_OF_syllables 3.VOWEL1_pos 4.VOWEL2_pos 5.VOWEL3_pos 6.VOWEL1
    #7VOWEL2 8.VOWEL3 9.p_s
    current = []
    count_vowel = 0
    vowels = [-1,-1,-1,-1]
    vowel_pos = [-1,-1,-1,-1,-1,-1,-1,-1]
#    vowel_pos2 = [-1,-1,-1,-1]

    for i in range(len(phonemes)):
        if phonemes[i] in vowel: 
            vowels[count_vowel]=vowel.index(phonemes[i])
            if i>0:
                vowel_pos[2*count_vowel] = words.index(phonemes[i-1])
            if i+1<len(phonemes):
                vowel_pos[1+2*count_vowel] = words.index(phonemes[i+1])
            count_vowel+=1
    has_pre = s_has_pre(vocab)
    has_end = s_has_end(vocab)
#    has_2 =0
#    has_3 =0
#    if count_vowel==2:
#        has_2 =1
#    if count_vowel==3:
#        has_3 =1
#    print()
#    if has_pre:
#        print(vocab)
#    print([count_vowel,count_vowel==2,count_vowel==3,count_vowel-3,has_pre,has_end])
#    r.append([count_vowel,has_2,has_3,count_vowel-3,has_pre,has_end])
#    r.append([count_vowel,len(phonemes)]+vowels+vowel_pos+vowel_pos2+[vocab,has_pre,has_end])
    r.append([count_vowel]+vowels+vowel_pos+[vocab,has_pre,has_end])


def get_Inf(s):
    dic = {}
    r = []
    vowel = "AA, AE, AH, AO, AW, AY, EH, ER, EY, IH, IY, OW, OY, UH, UW".replace(",","").split()
    consonant = "P, B, CH, D, DH, F, G, HH, JH, K, L, M, N, NG, R, S, SH, T, TH, V, W, Y, Z, ZH".replace(",","").split()
    for i in s:
        get_Slice(i,r,vowel,consonant)
    features_and_label = pd.DataFrame(r)
    return features_and_label

def get_Inf2(s):
    dic = {}
    r = []
    vowel = "AA, AE, AH, AO, AW, AY, EH, ER, EY, IH, IY, OW, OY, UH, UW".replace(",","").split()
    consonant = "P, B, CH, D, DH, F, G, HH, JH, K, L, M, N, NG, R, S, SH, T, TH, V, W, Y, Z, ZH".replace(",","").split()
    for i in s:
        get_Slice2(i,r,vowel,consonant)
    features_and_label = pd.DataFrame(r)
#    print(features_and_label)
    return features_and_label

def get_type(s):
    types = "CC,CD,DT,EX,FW,IN,JJ,JJR,JJS,LS,MD,NN,NNS,NNP,NNPS,PDT,POS,PRP,PRP$,RB,RBR,RBS,RP,SYM,TO\
    UH,VB,VBD,VBG,VBN,VBP,VBZ,WDT,WP,WP$,WRB".split(",")
    type_list = []
#    s = nltk.pos_tag(s)
#    print(s)
    for i in range (len(s)):
        word_type = nltk.pos_tag([s[i].capitalize()])
#        print(word_type)
        type_list.append(types.index(word_type[0][1]))
#    print(type_list)
    return type_list


################# training #################

def train(data, classifier_file):# do not change the heading of the function
    #train, test = train_test_split(data, test_size = 0.2)
    features_and_label = get_Inf(data)
    feature_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    features_and_label.loc[:,13] = get_type(features_and_label.loc[:,13])
    X_train = features_and_label[feature_list]
#    print(data[:5])
#    print(features_and_label.head())
#    print( X_train[10])
    y_train = features_and_label[16]
#    gnb = GaussianNB()
#    gnb.fit(X_train, y_train)
#    print(gnb.score(X_train,y_train))
#    depth = 0
#    for i in range(40,1,-1):
#        temp = evaluate(features_and_label,10,i)
#        if depth>temp:
#            print(i-1)
#            depth = i-1
#            break
#        else:
#            depth = temp
#    temp = evaluate(features_and_label,10,17)
    clf = DecisionTreeClassifier(criterion = "gini")
#    features_and_label2 = get_Inf(test)
    #X_test = features_and_label[feature_list]
    #X_test.loc[:,10] = get_type(X_test.loc[:,10])
    #y_test = features_and_label2[13]

#    clf = DecisionTreeClassifier(criterion = "gini")
    dtree = clf.fit(X_train, y_train)
    
    print(dtree.score(X_train,y_train))
    output = open('classifier_file', 'wb')
    pickle.dump(clf, output)
    output.close()
    return y_train    

################# testing #################

def test(data, classifier_file):# do not change the heading of the function
    pkl_file = open('classifier_file', 'rb')
    dt = pickle.load(pkl_file)
    r = []
    features_and_label = get_Inf2(data)
    features_and_label[13] = get_type(features_and_label[13])
    r = dt.predict(features_and_label)
    for i in range (len(r)):
        if r[i]==0:
            r[i]=1
    #print(r)
    pkl_file.close()
    return list(r)

def experiment(train, test, features, depth=5):
    X_train = train[features]
    y_train = train[len(features)]
    clf = DecisionTreeClassifier(criterion = "gini", max_depth = depth, random_state = 9)
    dtree = clf.fit(X_train, y_train)
    err_training = dtree.score(X_train,y_train)
    X_test = test[features]
    y_test = test[len(features)]
    err_testing = dtree.score(X_test,y_test)
    err_diff = err_training - err_testing
    print('{}, {}, {}'.format(err_training, err_testing, err_diff))
    return err_training, err_testing
        
def evaluate(data, repeat_times = 10, depth = 5):
    features = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    print('features: {}'.format(features))
    print('max_depth: {}\n'.format(depth))
    total_err_training = 0
    total_err_testing = 0
    for i in range(repeat_times):
        train_data, test_data = train_test_split(data, test_size = 0.33, random_state = 9 + i)
        err_training, err_testing = experiment(train_data, test_data, features, depth)
        total_err_training += err_training
        total_err_testing += err_testing
    return total_err_testing/repeat_times

#import helper
#import submission

#training_data = helper.read_data('./asset/review2.txt')
#classifier_path = './asset/classifier2.dat'
#train(training_data, classifier_path)
#test_data = helper.read_data('./asset/tiny_test.txt')
#prediction = test(test_data, classifier_path)
#print(prediction)
#from sklearn.metrics import f1_score
#ground_truth = [1, 1, 2, 1]
#print(f1_score(ground_truth, prediction, average='macro'))
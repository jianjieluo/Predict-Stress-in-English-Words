from submission import getInfoOfPronsFromTrain, getInfoFromTest
from pytest import raises


def test_getInfoOfPronsFromTrain():
    train1 = 'ENSINGER:EH1 N S IH0 N JH ER0'
    train2 = 'CANGELOSI:K AA0 NG G EH0 L OW1 S IY0'
    train3 = 'HADDOW:HH AE1 D OW0'
    train4 = 'DECONSTRUCTION:D IY0 K AH0 N S T R AH1 K SH AH0 N'
    train5 = 'TOYA:T OY1 AH0'

    f1, l1 = getInfoOfPronsFromTrain(train1.split(':')[0], train1.split(':')[1])
    f2, l2 = getInfoOfPronsFromTrain(train2.split(':')[0], train2.split(':')[1])
    f3, l3 = getInfoOfPronsFromTrain(train3.split(':')[0], train3.split(':')[1])
    f4, l4 = getInfoOfPronsFromTrain(train4.split(':')[0], train4.split(':')[1])
    f5, l5 = getInfoOfPronsFromTrain(train5.split(':')[0], train5.split(':')[1])   

    assert f1 == [6,9,7,-1,6,273009,272307,-1,3,491]
    assert l1 == 1

    assert f2 == [0,6,11,10,2400,282106,2511,3010,4,-1]
    assert l2 == 3

    assert f4 == [10,2,2,2,1810,2402,2730322902,243102,4,525]
    assert l4 == 3

    assert f5 == [12,2,-1,-1,3212,2,-1,-1,2,-1]
    assert l5 == 1


def test_getInfoFromTest():
    """
    PHONEMES = {'AA': 0, 'AE': 1, 'AH': 2, 'AO': 3, 'AW': 4, 'AY': 5, 'EH': 6, 'ER': 7,
    'EY': 8, 'IH': 9, 'IY': 10, 'OW': 11, 'OY': 12, 'UH': 13, 'UW': 14, 'P': 15,
    'B': 16, 'CH': 17, 'D': 18, 'DH': 19, 'F': 20, 'G': 21, 'HH': 22, 'JH': 23, 'K': 24,
    'L': 25, 'M': 26, 'N': 27, 'NG': 28, 'R': 29, 'S': 30, 'SH': 31, 'T': 32, 'TH': 33,
    'V': 34, 'W': 35, 'Y': 36, 'Z': 37, 'ZH': 38}
    """

    test1 = 'DATA:D EY T AH'
    test2 = 'MINING:M AY N IH NG'
    test3 = 'MACHINE:M AH SH IY N'
    test4 = 'LEARNING:L ER N IH NG'

    arr1 = getInfoFromTest(test1.split(':')[0], test1.split(':')[1])
    arr2 = getInfoFromTest(test2.split(':')[0], test2.split(':')[1])
    arr3 = getInfoFromTest(test3.split(':')[0], test3.split(':')[1])
    arr4 = getInfoFromTest(test4.split(':')[0], test4.split(':')[1])

    assert arr1 == [8,2,-1,-1,1808,3202,-1,-1,2,-1]
    assert arr2 == [5,9,-1,-1,2605,2709,-1,-1,2,308]
    assert arr3 == [2,10,-1,-1,2602,3110,-1,-1,2,-1]
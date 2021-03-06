
from lib2to3.pgen2.literals import test
import joblib
import os
import numpy as np
import random
import pickle
from scipy import signal as scipysignal
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from loguru import logger
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

def data_generate(trdata_path):
    # 读取数据
    pkl_file = open(os.path.join(trdata_path, 'data.pkl'), "rb")
    eeg = pickle.load(pkl_file)
    data = eeg['data'][:64,:]
    trigger = eeg['data'][-1,:]
    index = np.where(trigger != 0)[0]
    triggerdense = trigger[index]
    SRATE = eeg['srate']
    MICHS = np.array(range(17, 49))

    # left-mi 索引
    l_indx = index[triggerdense == 1]
    # left-mi 标签
    l_labels = triggerdense[triggerdense == 1]

    # right-mi 索引
    r_indx = index[triggerdense == 2]
    # right-mi 标签
    r_labels = triggerdense[triggerdense == 2]

    lr_indx = np.concatenate((l_indx, r_indx))
    # idle 标签
    i_labels = np.zeros_like(r_labels)+7
    # idle 索引
    i_indx = np.array(random.sample(list(lr_indx), len(i_labels))).astype(np.int32)

    data = data[MICHS,:]     #最后五个通道为心电通道等

    fs = SRATE / 2
    FLTNUM = scipysignal.firwin(SRATE * 3 + 1, np.array([7, 30]) / fs, pass_zero='bandpass')  # 带通滤波

    # 构造left-mi训练集
    l_epochdata = np.zeros([len(l_labels),data.shape[0],4 * SRATE])
    for i, point in enumerate(l_indx):
        epoch = data[:, point + int(0.5*SRATE):point + int(0.5*SRATE) + 4 * SRATE]  # 截取该段信号
        epoch = scipysignal.filtfilt(FLTNUM, 1, epoch, padlen=len(epoch)-1)
        l_epochdata[i, :, :] = epoch

    # 构造right-mi训练集
    r_epochdata = np.zeros([len(r_labels),data.shape[0],4 * SRATE])
    for i, point in enumerate(r_indx):
        epoch = data[:, point + int(0.5*SRATE):point + int(0.5*SRATE) + 4 * SRATE]  # 截取该段信号
        epoch = scipysignal.filtfilt(FLTNUM, 1, epoch, padlen=len(epoch)-1)
        r_epochdata[i, :, :] = epoch

    # 构造idle训练集
    i_epochdata = np.zeros([len(i_labels),data.shape[0],4 * SRATE])
    for i, point in enumerate(i_indx):
        epoch = data[:, point - int(4.5*SRATE): point - int(0.5*SRATE)]  # 截取该段信号
        epoch = scipysignal.filtfilt(FLTNUM, 1, epoch, padlen=len(epoch)-1)
        i_epochdata[i, :, :] = epoch
    print(i_epochdata.shape)

    return [l_epochdata, l_labels], [r_epochdata, r_labels], [i_epochdata, i_labels]

# def mi_train(data_list1,data_list2):
#     epochdata1 = data_list1[0]
#     epochdata2 = data_list2[0]
#     label1 = data_list1[1]
#     label2 = data_list2[1]
#     epochdata = np.concatenate((epochdata1,epochdata2),axis=0)
#     labels = np.concatenate((label1,label2),axis=0)

#     # csp共同空间模式
#     lda = LinearDiscriminantAnalysis()
#     csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

#     # 通过交叉验证来获得最佳的分类器
#     scores = []
#     traindataset = []
#     traindatasetid = []

#     #cv = ShuffleSplit(10, test_size=0.2)
#     cv = ShuffleSplit(10, test_size=0.2, random_state=42) 
#     cv_split = cv.split(epochdata)
#     for train_idx, test_idx in cv_split:
#         y_train, y_test = labels[train_idx], labels[test_idx]
#         try:
#             traindata = epochdata[train_idx]
#             X_train = csp.fit_transform(traindata, y_train)  # csp空间滤波器训练
#         except:
#             continue

#         lda.fit(X_train, y_train)  # 线性分类器训练
#         X_test = csp.transform(epochdata[test_idx])  # 测试集特征提取
#         scores.append(lda.score(X_test, y_test))
#         traindataset.append(traindata)
#         traindatasetid.append(y_train)

#     # 获得最佳的性能的分类器参数
#     if len(scores) > 0:  # 获得了分类器
#         mid = np.argsort(scores)[-1] # 返回元素值从小到大排序后索引值的数组, -1返回最大索引
#         X_train = csp.fit_transform(traindataset[mid], traindatasetid[mid])
#         lda.fit(X_train, traindatasetid[mid])

#         logger.info("============================================")
#         logger.info("最佳分类器性能")
#         logger.info("使用了{}组训练数据".format(epochdata.shape[0]))
#         logger.info("分类正确率为：{}".format(scores[mid]))
#         logger.info("============================================")
#         logger.info("socres: {}".format(scores))
#         logger.info("average scores {}".format(sum(scores)/len(scores)))

#         return csp, lda

#     else:
#         logger.info('更新分类器失败')


def mi_train(data_list1,data_list2):
    epochdata1 = data_list1[0]
    epochdata2 = data_list2[0]
    label1 = data_list1[1]
    label2 = data_list2[1]
    epochdata = np.concatenate((epochdata1,epochdata2),axis=0)
    labels = np.concatenate((label1,label2),axis=0)

    # 制作csp与svm的pipe
    pipe = Pipeline([('csp', CSP(n_components=4, reg=None, log=True, norm_trace=False)), ('svc', SVC())])

    # 通过交叉验证来获得最佳的分类器
    scores = []
    traindataset = []
    traindatasetid = []

    #cv = ShuffleSplit(10, test_size=0.2)
    cv = ShuffleSplit(10, test_size=0.2, random_state=42) 
    cv_split = cv.split(epochdata)
    for train_idx, test_idx in cv_split:
        y_train, y_test = labels[train_idx], labels[test_idx]
        try:
            X_train = epochdata[train_idx]
            pipe.fit(X_train, y_train)  # csp fit_transform, svm fit
        except:
            continue
        X_test = epochdata[test_idx]
        scores.append(pipe.score(X_test, y_test)) # csp transform, svm score
        traindataset.append(X_train)
        traindatasetid.append(y_train)

    # 获得最佳的性能的分类器参数
    if len(scores) > 0:  # 获得了分类器
        mid = np.argsort(scores)[-1] # 返回元素值从小到大排序后索引值的数组, -1返回最大索引
        pipe.fit(traindataset[mid], traindatasetid[mid]) # csp fit_transform, svm fit 用于保存模型

        logger.info("============================================")
        logger.info("最佳分类器性能")
        logger.info("使用了{}组训练数据".format(epochdata.shape[0]))
        logger.info("分类正确率为：{}".format(scores[mid]))
        logger.info("============================================")
        logger.info("socres: {}".format(scores))
        logger.info("average scores {}".format(sum(scores)/len(scores)))

        return pipe

    else:
        logger.info('更新分类器失败')


if __name__ == '__main__':
    # 数据、模型路径
    algr_path = os.path.abspath('..')
    mitrain_path = os.path.abspath('.')
    model_path = os.path.join(algr_path, 'MImodel')

    # 创建保存8个被试模型的文件夹
    num_subject = 8
    for i in range(num_subject):
        os.makedirs(model_path+'/S'+str(i+1), exist_ok=True)

    # 8个被试的数据路径
    trdata_path = []
    for i in range(num_subject):
        trdata_path.append(mitrain_path+'/Trainingdata/S'+str(i+1))

    # 获取8个被试的训练数据, 训练并保存模型
    for i in range(num_subject):
        l_list, r_list, i_list = data_generate(trdata_path[i])
        pipe_lr = mi_train(l_list, r_list)
        pipe_li = mi_train(l_list, i_list)
        pipe_ri = mi_train(r_list, i_list)
        joblib.dump(pipe_lr, model_path + '/S' + str(i+1) + '/pipe_lr.pkl')
        joblib.dump(pipe_li, model_path + '/S' + str(i+1) + '/pipe_li.pkl')
        joblib.dump(pipe_ri, model_path + '/S' + str(i+1) + '/pipe_ri.pkl')


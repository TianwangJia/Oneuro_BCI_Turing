import numpy as np
import os
import joblib
from scipy import signal as scipysignal
import warnings
warnings.filterwarnings("ignore")

rootdir = os.path.dirname(os.path.abspath(__file__))

class MImethod:
    def __init__(self, samp_rate):
        # 采样率
        self.samp_rate = samp_rate
        # 选择导联
        self.select_channel = range(17, 49)

    def pre_filter(self,data):
        # 选择导联
        data = data[self.select_channel, :]
        # 滤波
        fs = self.samp_rate / 2
        fltnum = scipysignal.firwin(self.samp_rate * 3 + 1, np.array([7, 30]) / fs, pass_zero='bandpass')  # 带通滤波
        filter_data = scipysignal.filtfilt(fltnum, 1, data, padlen=len(data) - 1)
        return filter_data

    def getmodel(self, sub_no):
        # 加载训练模型
        model_path = rootdir + '/MImodel/S'+ str(sub_no) + '/'  #每个被试单独一个模型
        #model_path = rootdir + '/MImodel/'
        # left-right
        self.pipe_lr = joblib.load(model_path + 'pipe_lr.pkl')
        # left_idle
        self.pipe_li = joblib.load(model_path + 'pipe_li.pkl')
        # right-idle
        self.pipe_ri = joblib.load(model_path + 'pipe_ri.pkl')

    def recognize(self, data, sub_no, roboaction):
        # 加载模型
        self.getmodel(sub_no)

        # 根据机器人动作分类，说明文件给出机器人动作编码
        # left - idle
        if roboaction in [1, 6, 8]:
            # 预测
            # proba = self.pipe_li.predict_proba(np.expand_dims(data, 0))
            result = self.pipe_li.predict(np.expand_dims(data, 0))

        # right - idle
        elif roboaction in [4, 7, 9]:
            # proba = self.pipe_ri.predict_proba(np.expand_dims(data, 0))
            result = self.pipe_ri.predict(np.expand_dims(data, 0))

        # left - right
        elif roboaction in [2, 3]:
        #    proba = self.pipe_lr.predict_proba(np.expand_dims(data, 0))
           result = self.pipe_lr.predict(np.expand_dims(data, 0))

        # 仅有SSVEP，不进行MI检测
        else:
            # proba = [[0, 0]]
            result = [7]

        # max(proba[0])
        return result[0]

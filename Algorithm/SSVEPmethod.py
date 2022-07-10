import numpy as np
from scipy import signal as scipysignal
from sklearn.cross_decomposition import CCA

class SSVEPmethod:
    def __init__(self, samp_rate):
        self.cca = CCA(n_components=1)
        # 采样率
        self.samp_rate = samp_rate
        # 选择导联
        self.select_channel = range(50,59)
        # 频率集合
        SSVEP_stim_freq = [round(15.4 + i * 0.1, 1) for i in range(4)]
        # 倍频数
        multiple_freq = 5
        # 参考信号时间
        templ_time = 5
        # 参考信号长度
        self.templ_len = templ_time * self.samp_rate
        # 正余弦参考信号
        self.target_template_set = []
        # 采样点
        samp_point = np.linspace(0, (self.templ_len - 1) / self.samp_rate, int(self.templ_len), endpoint=True)
        # (1 * 计算长度)的二维矩阵
        samp_point = samp_point.reshape(1, len(samp_point))
        # 对于每个频率
        for freq in SSVEP_stim_freq:
            # 基频 + 倍频
            test_freq = np.linspace(freq, freq * multiple_freq, int(multiple_freq), endpoint=True)
            # (1 * 倍频数量)的二维矩阵
            test_freq = test_freq.reshape(1, len(test_freq))
            # (倍频数量 * 计算长度)的二维矩阵
            num_matrix = 2 * np.pi * np.dot(test_freq.T, samp_point)
            cos_set = np.cos(num_matrix)
            sin_set = np.sin(num_matrix)
            cs_set = np.append(cos_set, sin_set, axis=0)
            self.target_template_set.append(cs_set)

    # 识别算法
    def recognize(self, data):
        p = []
        data = data.T
        # 对每个频率
        for template in self.target_template_set:
            # 参考信号
            template = template[:, 0:data.shape[0]]
            template = template.T
            # 计算相关系数
            self.cca.fit(data,template)
            data_tran, template_tran = self.cca.transform(data,template)
            rho = np.corrcoef(data_tran[:,0],template_tran[:,0])[0,1]
            p.append(rho)
        result = p.index(max(p))
        result = result + 3
        p.sort(reverse=True)
        if p[0] > 0.5:
            return result, max(p)
        else:
            return 7, max(p)

    # 预处理
    def pre_filter(self,data):
        # 选择导联
        data = data[self.select_channel, :]
        # 滤波
        f0 = 50
        q = 35
        b, a = scipysignal.iircomb(f0, q, ftype='notch', fs=self.samp_rate)
        fs = self.samp_rate / 2
        N, Wn = scipysignal.ellipord([6 / fs, 90 / fs], [2 / fs, 100 / fs], 3, 40)
        b1, a1 = scipysignal.ellip(N, 1, 40, Wn, 'bandpass')
        filter_data = scipysignal.filtfilt(b1, a1, scipysignal.filtfilt(b, a, data))
        return filter_data
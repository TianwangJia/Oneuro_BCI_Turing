from Algorithm.Interface.AlgorithmInterface import AlgorithmInterface
from Algorithm.Interface.Model.ReportModel import ReportModel
from Algorithm.SSVEPmethod import SSVEPmethod
from Algorithm.MImethod import MImethod
import numpy as np
from loguru import logger

'''仅用于调试程序'''

class AlgorithmImplementTuring(AlgorithmInterface):
    # 类属性：范式名称
    PARADIGMNAME = 'Turing'

    def __init__(self):
        super().__init__()
        # 定义采样率，题目文件中给出
        self.samp_rate = 250
        # trial开始trigger，题目说明中给出
        self.trial_start_trig = 240
        # 计算时间
        cal_time = 3
        # 计算长度
        self.cal_len = 600  #cal_time * self.samp_rate  #750  600,500
        # cache初始化
        self.init_cache()
        # SSVEP算法
        self.ssvep_method = SSVEPmethod(self.samp_rate)
        # MI算法
        self.mi_method = MImethod(self.samp_rate)

    def init_cache(self):
        self.eeg_cache = np.zeros((64, 0), dtype='float64')
        self.cur_roboaction = None
        self.bcitask_startpoint = float('inf')

    def run(self):
        # 停止标签,由框架给出
        end_flag = False
        while not end_flag:
            data_model = self.task.get_data()
            self.process(data_model)
            end_flag = data_model.finish_flag

    def process(self, data_model):
        # eeg信号
        eeg = data_model.data[:-1,:]
        # trigger
        tri = data_model.data[-1,:]
        # 一个数据包的长度
        N = eeg.shape[1]
        # 数据拼接
        self.eeg_cache = np.hstack((self.eeg_cache, eeg))

        # 搜索trial开始数据包
        if np.where(tri == self.trial_start_trig)[0].size > 0:
            self.eeg_cache = eeg
            # 初始化时设置一个较大的数
            self.bcitask_startpoint = float('inf')
            logger.info("[user debug] new trial")

        # 搜索roboaction trigger，同时也是bci任务的起点
        tri_id = np.where((tri > 0) & (tri < 240))[0]
        if tri_id.size > 0:
            tri_id = tri_id[0]
            # roboaction占位在3-6位
            self.cur_roboaction = int(tri[tri_id]) >> 3
            self.cur_bci_task = int(tri[tri_id]) & 0b00000111
            # bcitask起始点在eeg_cache中的位置
            self.bcitask_startpoint = self.eeg_cache.shape[1] - N + tri_id
            logger.info("[user debug] robot action: {}".format(self.cur_roboaction))

        # 累计了足够的数据，进入信号处理
        if self.eeg_cache.shape[1] - self.bcitask_startpoint >= self.cal_len:
            result = self.recognize(self.eeg_cache, self.bcitask_startpoint, data_model, self.cur_roboaction)
            report_model = ReportModel()
            report_model.result = result
            # 报告结果
            self.task.report(report_model)
            # 释放数据
            self.init_cache()

    def recognize(self, data, startpoint, data_model, roboaction):  #最重要,robaaction
        # 信号预处理
        datafilter_ssvep = self.ssvep_method.pre_filter(data)
        # 用于分类的数据
        recdata_ssvep = datafilter_ssvep[:, startpoint:startpoint + self.cal_len]
        # 信号预处理
        datafilter_mi = self.mi_method.pre_filter(data)
        # 用于分类的数据
        recdata_mi = datafilter_mi[:, startpoint:startpoint + self.cal_len]
        # 分类结果
        result_ssvep, proba_ssvep = self.ssvep_method.recognize(recdata_ssvep)
        # result_mi, proba_mi = self.mi_method.recognize(recdata_mi, data_model.subject_id, roboaction)
        result_mi = self.mi_method.recognize(recdata_mi, data_model.subject_id, roboaction)
        # 结果分析
        if result_ssvep < 7:
            result = result_ssvep
        else:
            result = result_mi
        if roboaction == 5:
            result = result_ssvep
        return result
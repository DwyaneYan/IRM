import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import scipy
import time
import copy

# -*- coding: UTF-8 -*-
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

## region 指定超参数
VlistStart = 4
Vmin = 2.5
Vmax = 3.7
Soc_av = 15
TimeWin_Corr = 600
TimeJumpStep = 120
CutTimeWnd = 300
CutTimeWnd_DistbPul = 600
DiffTime = 300
MoothTimeWnd = 1800
SpeedWndTime = 900
TimeWnd_p = 1800
SOC_min = 30
SOC_max = 80
TimeWnd_Sp = 1800
NormalPartRatio = 0.7
LowPercent = 0.6
HighPercent = 0.8
MinRiskIta = 20
MinRiskQ = 0.0005


## endregion


def T1(RiskMark, r1, Q1, t, V00, p, dt0, Soc, I, z, filename, bc, Ita, Q):
    """
    绘制高风险点细节图
    :param RiskMark: 高风险标记位置标记
    :param r1: 风险特征
    :param Q1: 相对风险值
    :param t: 时间
    :param V00: 单体电压
    :param p: 干扰脉冲标记
    :param dt0: 时间不连续标记
    :param Soc: Soc
    :param I: 电流
    :param z: 单体中值压差
    :param filename: vin
    :param bc: 高风险点标记位置
    :param Ita: 绝对风险
    :param Q: 相对风险
    :return:
    """
    LineNum = 3
    MarkNum = sum(RiskMark)
    dt0 = np.array(dt0)
    V00 = np.array(V00)
    I = np.array(I)
    Soc = np.array(Soc)

    if MarkNum == 0:
        print('低风险车')
    if MarkNum <= LineNum:
        LineNum = MarkNum
    Nm = int(MarkNum // LineNum)
    if LineNum * Nm < MarkNum:
        Nm += 1

    Nr = len(RiskMark)

    # 取全部的风险点
    bc = np.sort(bc)
    r2 = copy.deepcopy(r1)
    r2[RiskMark == 0] = 0
    CurIta = np.sqrt(r1 + 0.000001) / (Q1 + 0.000001)

    RiskLine = np.zeros((Nr, 1))
    RiskLine[CurIta > 2.5] = 1
    RiskLine[r1 <= 0.00025] = 0
    RiskLine = RiskLine + 2.5
    dr = np.diff(RiskLine)
    dr[Nr - 1] = 0
    LineNum = int(LineNum)
    for i in range(Nm):
        for j in range(LineNum):
            fig = plt.figure(filename)
            km = i * LineNum + j
            if km > MarkNum:
                break
            indec_bc = bc[km]
            minPos = max(0, indec_bc - 1000 + 1)
            maxPos = min(indec_bc + 1000 + 1, Nr - 1)
            k1 = np.arange(minPos, maxPos)

            V0 = V00[k1, :]
            Evs = p[k1]
            ct = dt0[k1]
            SocSlice = Soc[k1]
            I = I[:, np.newaxis]
            ISlice = I[k1]
            RLine = RiskLine[k1]
            V1 = z[k1]
            c = sum(pow(V1[500:min(1500, V1.shape[0] + 1)], 2))  # 按列求和
            b1 = np.argsort(-c)
            d1 = np.sort(-c)
            # 数据准备好了，开始画图
            str_time = time.localtime(int(t[bc[km]]))
            standard_d = time.strftime("%Y-%m-%d %H:%M:%S", str_time)
            Va = np.median(V0, axis=1)
            str_title = '%s:%d; Ita=%.0f,Q=%.2f; %s=%d;\n电芯:%d,%d,%.1f; SOC=%.0f,Vm=%.1f,I=%.0f; \nCI=%.0f,Cq=%.2f' % (
                filename, Nm, Ita, Q / 0.0005, standard_d, bc[km], b1[1], b1[2], d1[1] / d1[2], Soc[bc[km]], Va[1000],
                I[bc[km]][0], CurIta[bc[km]][0], r1[bc[km]][0] / 0.0005)
            ax1 = fig.add_subplot(211)

            # 设置字体为楷体
            # plt.rcParams['font.sans-serif'] = ['KaiTi']

            plt.title(str_title)
            ax1.plot(V0)
            ax1.plot(V0[:, b1[1]], 'g', linewidth=2)
            ax1.plot(V0[:, b1[2]], 'r', linewidth=2)
            ax1.plot(RLine, 'r')
            ax1.set_ylim(2.5, 3.5)
            ax12 = ax1.twinx()
            ax12.plot(ct * 0.25, 'b')
            ax12.plot(SocSlice / 100, 'k')
            ax12.plot(Evs)
            ax12.set_ylim(0, 1.5)

            ax2 = fig.add_subplot(212)
            plt.rcParams['font.sans-serif'] = ['KaiTi']
            ax2.plot(V1)
            ax2.plot(V1[:, b1[1]], 'g', linewidth=2)
            ax2.plot(V1[:, b1[2]], 'r', linewidth=2)
            ax22 = ax2.twinx()
            ISlice = np.squeeze(ISlice)
            ax22.plot(ISlice, 'b')
            ax22.set_ylim([-150, 500])
            plt.show()
            plt.savefig("result/%s_%d.png" % (filename, j))
            plt.close()


def eachFile(filePath):
    '''
    遍历目录下所有文件并返回文件名列表及文件地址列表
    :param filePath:
    :return:
    '''
    fileList = []
    fileNameList = os.listdir(filePath)
    fileNameList = [x.split('.')[0] for x in fileNameList]
    for fileName in fileNameList:
        file = os.path.join('%s\\%s.csv' % (filePath, fileName))
        fileList.append(file)
    return fileNameList, fileList


def ExtractData(df):
    '''
    解析单体电压
    :param df:单体电压以|隔开
    :return:  TIME,CHARGE_STATUS,SUM_CURRENT,SOC,1,2...格式的DataFrame
    '''
    df.drop(df[df['CELL_VOLT_LIST'] == '65535'].index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    tmp = df['CELL_VOLT_LIST'].values
    df.drop(['CELL_VOLT_LIST'], axis=1, inplace=True)
    single_V = []
    for i in range(len(tmp)):
        try:
            if len(tmp[i]) < 50:
                df = df.drop([i])
                df.reset_index(drop=True, inplace=True)
                continue

            b = tmp[i].split('|')
            b = [int(c) for c in b]
            single_V.append(b)
        except:
            df.reset_index(drop=True, inplace=True)
    del tmp
    df = pd.concat([df, pd.DataFrame(single_V)], axis=1)
    return df


def GetData(filepath):
    '''
    组织数据，SampleData格式
    :param filepath:
    :return:
    '''
    # df = pd.read_csv(filepath, usecols=["TIME", "CHARGE_STATUS", "SUM_CURRENT", "SOC", "CELL_VOLT_LIST"], nrows=10000)
    df = pd.read_csv(filepath, usecols=["TIME", "CHARGE_STATUS", "SUM_CURRENT", "SOC", "CELL_VOLT_LIST"])
    df = ExtractData(df)
    return df


def GetFs(t):
    '''
    计算采样频率
    :param t: df
    :return:
    '''
    dt = t.diff()
    dt[dt > 60] = 60
    dt[dt < 10] = 10
    dv = dt.mean()
    Fs = 1 / dv
    Fsd = np.array([1, 1 / 10, 1 / 20, 1 / 30, 1 / 60])
    ds = np.abs(Fsd - Fs)
    idx = list(ds).index(min((ds)))
    Fs = Fsd[idx]
    return Fs


def FastCleanVData(V, Vmax, Vmin):
    '''
    上下限清洗
    :param V:
    :param Vmax:
    :param Vmin:
    :return:
    '''
    [m, n] = V.shape

    VMa = V.max(axis=1)
    VMi = V.min(axis=1)
    w = (VMa > Vmax) | (VMi < Vmin)
    Vm = V.median(axis=1)
    Vd = V.diff(axis=0)
    Vd.at[0, :] = 0
    Vd = np.abs(Vd)
    Vd = Vd.max(axis=1) - Vd.min(axis=1)
    w = (Vd > 0.25) | w

    # Todo  用矩阵相乘替换异常值为中位数
    # V = np.dot(w, Vm) + np.dot((1 - w), V)  ## 待解决 Unable to allocate 70.8 GiB for an array with shape (97451, 97451) and data type float64

    for i in range(len(w)):
        if w[i]:
            V.at[i, :] = Vm[i]
    return V


def CleanDatabyVoltageList(data, VlistStart, VlistEnd, Vmin, Vmax):
    '''
    数据清洗
    :param data: df,最少包括时间戳数据
    :param VlistStart: 单体电压开始列索引
    :param VlistEnd: 单体电压结束列索引
    :param Vmin: 电压清洗上限
    :param Vmax: 电压清洗下限
    :param FreqAdjust: 采样频率矫正标志位，为1则矫正，默认为0.1Hz，低于0.1Hz的通过线性插值补充为0.1Hz
    :return:
    '''
    t = data.iloc[:, 0]  # 取时间列
    Fs = GetFs(t)
    TimeStep = t.diff()
    TimeStep[0] = 10
    if TimeStep.min() == 0:
        data.drop(list(TimeStep[TimeStep == 0].index), inplace=True)
        data.reset_index(drop=True, inplace=True)
        t = data.iloc[:, 0]
        TimeStep = t.diff()
        TimeStep[0] = 10
    if TimeStep.min() < 0:
        print('err: data need time sort')
        data.sort_values(by='TIME', inplace=True)
        data.reset_index(drop=True, inplace=True)
        t = data.iloc[:, 0]
    V = data.iloc[:, VlistStart:VlistEnd] / 1000
    V = FastCleanVData(V, Vmax, Vmin)
    Vm = V.max(axis=1)
    V = V.loc[Vm[Vm < 10].index]
    data = data.loc[Vm[Vm < 10].index]
    Vm = V.min(axis=1)
    V = V.loc[Vm[Vm > 0.1].index]
    data = data.loc[Vm[Vm > 0.1].index]
    # V = fc.FastCleanVData(V, Vmax, Vmin)
    V[V < Vmin] = Vmin
    V[V > Vmax] = Vmax
    data.iloc[:, VlistStart:VlistEnd] = V
    [N, M] = V.shape
    return data, Fs, N, M


def GetAverV(x, TimeWnd, Fs):
    '''
    均值滤波
    :param x: 信号数据
    :param TimeWnd: 时间窗(s)
    :param Fs: 采样频率
    :return:
        u: 单体电压均值滤波后的信号，与原信号相同维度
    '''
    try:
        N, m = x.shape
    except:
        N = x.shape[0]
        m = 1
    x = np.array(x)

    Sp = np.zeros((N, m))
    if m == 1:
        e = 0
    else:
        e = np.zeros((1, m))

    for k in range(N):
        e = e + x[k]
        Sp[k] = e  # # Voila积分
    KK = int(np.fix((Fs * TimeWnd) / 2))
    K = int(KK * 2 + 1)
    r = np.zeros((N - K, m))
    for k in range(N - K):
        r[k] = (Sp[k + K] - Sp[k]) / K;  # 均值  =  时间窗内（x[k+1] 到  x[k+K]） 的和，然后除以时间窗长度K
    u = np.zeros((N, m))
    u[:KK] = (np.ones((KK, 1)) * r[0, :]).reshape(KK, m)
    u[KK:N + KK - K] = r
    u[N + KK - K:] = (np.ones((KK + 1, 1)) * r[-1, :]).reshape(KK + 1, m)
    return u


def GetIandVCorr(data, TimeWnd, Fs):
    '''
    通过计算电压电流相关性清洗数据
    :param data: 待清洗数据
    :param TimeWnd: 时间窗口
    :param Fs: 采样频率
    :return: 清洗后data, 单体电压中位数V
    '''
    V = np.median(data.iloc[:, 4:], axis=1)
    Vv = GetAverV(np.square(V), TimeWnd, Fs) - np.square(GetAverV(V, TimeWnd, Fs))
    Vv[Vv < 0] = 0
    Id = np.diff(data.SUM_CURRENT)
    Id = np.append(Id, Id[-1])
    Vd = np.diff(V)
    Vd = np.append(Vd, Vd[-1])
    Vs = abs(Id * Vd)
    n = len(data.SUM_CURRENT)
    Vs = GetAverV(Vs, TimeWnd, Fs)
    Cr = np.ones(n)

    b1 = (Vs < 1e-6).reshape(len(Vs))
    b2 = abs(data.SUM_CURRENT) > 0
    b3 = data.CHARGE_STATUS != 1
    b4 = (Vs == 0).reshape(len(Vs))
    b5 = data.SUM_CURRENT == 0
    b6 = (Vv == 0).reshape(len(Vv))
    b7 = data.SUM_CURRENT < -5
    b8 = data.CHARGE_STATUS == 1

    Cr[(b1 & b2 & b3) | (b4 & b5) | (b6 & b7 & b8)] = 0
    Cr = GetAverV(Cr, TimeWnd, Fs)
    Cr[Cr > 0.9] = 1
    Cr[Cr <= 0.9] = 0
    data = data.iloc[Cr == 1]
    data.reset_index(drop=True, inplace=True)

    return data, V


def GetSocFilter(signal, threshold):
    '''
    SOC滤波
    :param signal: soc信号数据
    :param threshold: 阈值
    :return: 清洗后的SOC:signal
    '''
    n = len(signal)
    socDiff = abs(np.diff(signal))
    socDiff[0] = 0
    st, et, ms = 0, 0, 0
    signal = np.array(signal)
    for i in range(2, n - 1):
        if socDiff[i] < threshold and (
                (signal[i - 1] < 15) or (signal[i] < 15) or (signal[i + 1] < 15) and ms == 0 and et == 0):
            sc1 = signal[i]
            st = i
            continue
        if socDiff[i] >= threshold and ms == 0 and st > 0:
            et = i
            ms = -1
            continue
        if socDiff[i] < threshold and st > 0 and et > 0 and ms == -1:
            et = i
        if socDiff[i] >= threshold and ms == -1 and st > 0:
            et = i
            ms = 1
            continue
        if socDiff[i] < threshold and ms == 1:
            sc2 = signal[i]
            et = i
            kk = et - st + 1
            z = np.arange(kk - 1)
            if abs(sc2 - sc1) < 15:
                signal[st:et] = sc1 + z / kk * (sc2 - sc1)
                temp = i
            st, et, ms = 0, 0, 0
        if (ms == -1 or ms == 1) and abs(signal[i] - sc1) > 16 and socDiff[i] > 0:
            st, et, ms = 0, 0, 0
    return signal


def GetTimeJumpMark(t, Fs, TimeJumpStep, CutTimeWnd):
    '''
    识别时间不连续的位置并去掉跳跃窗口的位置
    :param t: 时间序列
    :param Fs: 采样频率
    :param TimeJumpStep: 判断为时间跳跃的阈值
    :param CutTimeWnd: 去掉的时间窗口长度
    :return:时间不连续干扰片段标记向量dt, 时间不连续的位置标记向量dt0
    '''
    dt = list(t.diff())
    dt = pd.DataFrame(dt)
    dt.iloc[0] = dt.iloc[1]
    dt[dt <= TimeJumpStep] = 1
    dt[dt > TimeJumpStep] = 0
    dt.iloc[0, 0] = dt.iloc[1, 0]
    dt0 = dt  # copy.deepcopy(np.array(dt))
    if CutTimeWnd > 10:
        dt = GetAverV(1 - dt, CutTimeWnd, Fs)
        dt[dt > 0] = 1
        dt = 1 - dt
    return dt, dt0


def GetAverSpeed(x, TimeWnd, Fs):
    '''
    速度滤波
    :param signal:
    :param timeWnd:时间窗口
    :param Fs:采样频率
    :return: 滤波后信号u
    '''
    N, m = x.shape
    KK = int(np.fix(np.fix(Fs * TimeWnd) / 2 + 0.5));
    K = int(2 * KK + 1)
    v = np.zeros((N - K, m))
    for k in range(N - K):
        v[k, :] = (x[k + K, :] - x[k, :]) / (K / Fs)
    u = np.zeros((N, m))
    u[:KK] = (np.ones((KK, 1)) * v[0, :]).reshape(KK, m)
    u[KK:N + KK - K] = v
    u[N + KK - K:] = (np.ones((KK + 1, 1)) * v[-1, :]).reshape(KK + 1, m)
    return u


def GetDistbPulsePos(signal, Fs, CutTimeWnd, DiffTime):
    '''
    获取不连续的脉冲干扰，如信号过度区域，信号失真的位置
    :param signal: 单个信号
    :param Fs:采样频率
    :param CutTimeWnd: 判定干扰影响范围的窗口
    :param DiffTime: 均值滤波窗口
    :return: 受干扰影响的标记p1
    '''
    N0 = len(signal)
    signal = GetAverV(signal, DiffTime, Fs)
    df = GetAverSpeed(signal, 60, Fs)
    sd = 3 * df.std()
    p1 = np.zeros((N0, 1))
    p1[abs(df) > sd] = 1
    p1 = GetAverV(p1, CutTimeWnd, Fs)
    p1[p1 > 0] = 1
    return p1


def GetHighSpeed(signal, Fs, MoothTimeWnd, SpeedWndTime):
    '''
    提取安全要素
    :param signal:单体电压
    :param Fs:采样频率
    :param MoothTimeWnd:均值滤波窗口
    :param SpeedWndTime:速度滤波窗口
    :return:安全要素Vv
    '''
    signal = GetAverV(signal, MoothTimeWnd, Fs)
    Vv = GetAverSpeed(signal, SpeedWndTime, Fs) * SpeedWndTime
    Vv[Vv > 10] = 10
    Vv[Vv < -10] = -10
    Vv = np.exp(0.2 * Vv)
    return Vv


def CellsAverMultiRatio(V, TimeWnd, Fs):
    '''
    基于方差熵的能量一致性的安全量化方法
    :param V: 单体电压
    :param TimeWnd: 时间窗（s）
    :param Fs: 采样频率
    :return:
        Ita:N*1维的安全量化值
    '''
    c = np.square(GetAverV(V, TimeWnd, Fs))  # 均值的平方
    d = GetAverV(np.square(V), TimeWnd, Fs)  # 平方的均值

    c = np.square(np.mean(c, axis=1)) + 1e-8
    d = np.mean(np.square(d), axis=1) + 1e-8

    Ita = c / d
    Ita[Ita > 1] = 1
    Ita = Ita.reshape(len(Ita), 1)
    return Ita


def GetRiskPointInfo(Risk, RefRisk, RiskWnd, Fs):
    '''

    :param Risk: 风险特征
    :param RefRisk: 风险特征的阈值
    :param RiskWnd:
    :param Fs:
    :return: 高风险值Q； 高风险点标记：RiskMark
    '''
    K = int(np.fix(RiskWnd / 2 * Fs))
    z = Risk / RefRisk
    n = len(Risk)
    mark = np.zeros((n, 1))
    RiskMark = np.zeros((n, 1))
    z1 = z
    z1[z1 < 1] = 0
    b = find_peaks(z1[:, 0], width=K)[0]  # 找到极值点的位置索引，放在一个数组b里
    # todo  排序，取最大的15个
    Np = len(b)
    for k in range(Np):
        b1 = np.arange(b[k] - K, b[k] + K + 1)
        b1 = b1[b1 >= 1]
        b1 = b1[b1 <= n]
        mark[b1] = 1
    Q = np.sqrt(np.mean(np.square(Risk[mark == 1])))  # 所有高风险点平方、均值、再开方
    RiskMark[b] = 1
    return Q, RiskMark


def GetRiskAssesmentData(p, Fs, TimeWnd, NormalPartRatio, LowPercent, HighPercent, MinRiskIta, MinRiskQ, RiskOutMode):
    '''
    风险累计及状态感知
    :param p: 风险量化时间序列
    :param Fs:采样频率
    :param TimeWnd:时间窗口
    :param NormalPartRatio: 在已有数据中假设为正常态的比例
    :param LowPercent:  去掉正常态数据以后，低风险数据的比例
    :param HighPercent:  去掉正常态数据以后，高风险数据的比例
    :param MinRiskIta: Ita的下限
    :param MinRiskQ: Q的下限
    :param RiskOutMode: 为1时，输出Ita与Q，用于风险预警的同时，表示还需要输出风险点信息用于风险分析定位；为0时，只输出Ita与Q
    :return:
        绝对风险Ita, 风险状态(Ita值分子)Q, 稳定状态(Ita值分母)Q1, 风险累计值Sp, 最小风险位置minPos, 最大风险位置maxPos, 风险点标记RiskMark, 正常点标记NormalMark, 风险量化特征r1
    '''
    if RiskOutMode == 1 and (MinRiskIta * MinRiskQ) == 0:
        MinRiskIta = 5
        MinRiskQ = 0.00005
    N = len(p)
    Sp = np.zeros((N, 1))
    if LowPercent == 0:
        LowPercent = 0.6
    if HighPercent == 0:
        HighPercent = 0.8
    e = 0
    for k in range(N):
        e = e + p[k]
        Sp[k] = e
    Sp = Sp / Fs
    r = GetAverSpeed(Sp, TimeWnd, Fs)  ## 计算风险量化特征
    r1 = r
    Nr = len(r)
    z = np.square(r)
    a = np.sort(z[:, 0])
    Nc = int(np.fix(Nr * NormalPartRatio))  ## 排序后，判定为正常点的个数
    a = a[Nc - 1:]
    Nr = len(a)
    N1 = int(np.fix(Nr * LowPercent))

    Riska = GetAverV(p, TimeWnd, Fs)
    z1 = z
    z1[Riska < 0.00005] = 0  # z1为将正常风险置0后的风险特征
    Q1 = np.mean(a[1:N1])
    Q, RiskMark = GetRiskPointInfo(z1, MinRiskQ, TimeWnd, Fs)

    Ita = np.sqrt((Q + 0.000001) / (Q1 + 0.000001))
    z = np.square(r1 - np.sqrt(Q1))
    k1 = np.argmin(z)
    minPos = k1  ## 寻找风险较小且稳定的位置
    z = np.square(r1 - np.sqrt(Q))
    k2 = np.argmax(z)
    maxPos = k2  ## 寻找风险较大且稳定的位置
    NormalMark = np.zeros((N, 1))
    if RiskOutMode == 1:
        NormalMark[r1 <= np.sqrt(Q1)] = 1
        r1 = np.square(r1)
    else:
        NormalMark = 0
        r1 = 0
    return Ita, Q, Q1, Sp, minPos, maxPos, RiskMark, NormalMark, r1


def getRangeOfRiskSample(N, V, pos, Vs, dt1, dt0):
    """
    取风险点的上下1000数据，并且丢弃异常下标，通过获得的下标取电压和dt
    :param pos:
    :param Vs: 原始电压
    :param dt1: 原始dt1
    :param dt0: 原始dt0
    :return:
    """
    minRange = max(0, pos - 1000)
    maxRange = min(N, pos + 1000)
    k1 = np.arange(minRange, maxRange)
    newVs = V.loc[minRange:maxRange]
    newDt1 = dt0.loc[minRange:maxRange]
    Vs = Vs.append(newVs, ignore_index=True)
    dt1 = dt1.append(newDt1, ignore_index=True)
    return k1, Vs, dt1


def GetSlide(Fs, Sp):
    """
    通过Fs取步长，对Sp降采样
    :param Fs:
    :param Sp:
    :return:
    """
    if Fs == 0.1:
        slide = 12
    else:
        slide = 4
    SpRiskSample = []
    for i in range(1, len(Sp)):
        if i % slide == 0:
            SpRiskSample.append(Sp[i])
    return SpRiskSample


def GetMoment(p, Fs):
    p = np.array(p)
    N, m = p.shape
    Sp = np.zeros((N, m))
    e = Sp[0]
    for i in range(N):
        e += p[i]
        Sp[i] = e
    Sp /= Fs
    MIta = max(Sp[-1]) / np.median(Sp[-1])
    return Sp, MIta


def TDraw(V, r1, RiskMark, Soc, z, Sp1, Sp, filename, Ita, Q):
    V = np.array(V)
    # 画图3要用的数据
    Nm = int(sum(RiskMark)[0])
    a = np.argsort(RiskMark)
    index = np.arange(len(RiskMark))
    b = np.transpose(np.nonzero(RiskMark))[:, 0]
    K1 = min(b) - 20000
    K2 = max(b) + 20000
    kc = np.arange(K1, K2)
    y = np.ones((1, 1))
    RiskMark1 = scipy.signal.convolve(RiskMark, y)
    # RiskMark1 = RiskMark1[:-400]
    RiskMark1[RiskMark1 >= 0.0001] = 1
    RiskMark1[RiskMark1 < 0.0001] = 0
    Sp1 = np.array(Sp1)
    b1 = np.argsort(-Sp1[-1, :])
    str_title1 = '%s,Ita = %.0f,Q=%.2f,电芯 %d,%d' % (filename, Ita, Q / 0.005, b1[0] + 1, b1[1] + 1)
    # str_title2 = '%s,Ita = %.0f,Q=%.2f,电芯 %d,%d' % (filename, Ita, Q / 0.005, b1[0] + 1, b1[1] + 1)

    fig = plt.figure(str_title1)
    # 第一个图
    ax1 = fig.add_subplot(111)
    plt.title(str_title1)
    RiskLine = r1 > 0.0005
    ax1.plot(V)
    ax12 = ax1.twinx()
    ax12.plot(RiskLine, 'r', linewidth=1)
    ax12.set_ylim(0, 6)
    plt.savefig("result/%s.png" % filename)
    plt.close()

    return b


def GetSampleDataReady(r1, V, Fs, Sp, RiskMark, Ita1, Q, dt, p1, Soc, filename):
    """
    绘制高风险点位置标记图
    :param r1: 风险量化特征
    :param V: 单体电压
    :param Fs: 采样频率
    :param Sp: 风险累计值
    :param RiskMark: 风险标记向量
    :param Ita1: 绝对风险
    :param Q: 相对风险
    :param dt: 时间不连续标记
    :param p1: 干扰脉冲标记
    :param Soc: Soc
    :param filename: vin
    :return:中值压差z, 风险点个数R1, 风险点位置索引b
    """
    Sp = GetSlide(Fs, Sp)
    R1 = sum(RiskMark)
    Vm = V.median(axis=1)
    z = np.array(V)
    Vm = np.array(Vm)
    Vm = Vm[:, np.newaxis]

    z = z - Vm * np.ones((1, z.shape[1]))  # 中值压差
    Vv1 = z * z
    dt = np.array(dt)
    dt = dt * np.ones((1, z.shape[1]))
    Vv1[dt == False] = 0
    p1 = p1 * np.ones((1, z.shape[1]))
    Vv1[p1 == True] = 0

    Sp1, MIta1 = GetMoment(Vv1, Fs)
    Sp1 = GetSlide(Fs, Sp1)

    # 开始画图
    b = TDraw(V, r1, RiskMark, Soc, z, Sp1, Sp, filename, Ita1, Q)
    return z, R1, b


if __name__ == '__main__':
    filelist, filepathlist = eachFile(r'D:\OneDrive\data\futian_testdata\data')
    for i in range(len(filelist)):
        filename = filelist[i]
        print("==============> %d %s <===================" % (i, filelist[i]))
        # todo 获取数据
        df = GetData(filepathlist[i])
        ## todo 数据清洗
        df.dropna(axis=0, inplace=True)
        df.drop(df[df.SOC > 100].index, inplace=True)
        df.reset_index(drop=True, inplace=True)
        VlistEnd = df.shape[1]
        data, Fs, N, M = CleanDatabyVoltageList(df, VlistStart, VlistEnd, Vmin, Vmax)
        ## 电压电流相关性判断
        data, Vm = GetIandVCorr(data, TimeWin_Corr, Fs)

        t = data.iloc[:, 0]
        cs = data.iloc[:, 1]
        I = data.iloc[:, 2]
        Soc = data.iloc[:, 3]
        V = data.iloc[:, 4:]

        ## todo 异常处理
        ## SOC滤波
        Soc = GetSocFilter(Soc, Soc_av)
        ## 时间跳跃点判断
        dt, dt0 = GetTimeJumpMark(t, Fs, TimeJumpStep, CutTimeWnd)
        ## 干扰脉冲判断
        p1 = GetDistbPulsePos(V.max(axis=1) - V.min(axis=1) + V.median(axis=1), Fs, CutTimeWnd_DistbPul, DiffTime)
        dt = dt == 1
        p1 = p1 == 1
        w = dt & p1
        sw = Soc < SOC_min
        sw2 = Soc > SOC_max
        sw = sw | sw2
        w = w.reshape(len(w)) | sw

        ## todo 模型计算
        ## 提取安全要素
        Vv = GetHighSpeed(250 * np.square(((V.T - V.T.median()).T)) + np.square(V + 0.35), Fs, MoothTimeWnd,
                          SpeedWndTime)
        ## 安全特征量化
        p = CellsAverMultiRatio(Vv, TimeWnd_p, Fs)
        p[w] = 1

        ## 风险累计
        Ita1, Q, Q1, Sp, minPos, maxPos, RiskMark, NormalMark, r1 = GetRiskAssesmentData(1 - p, Fs, TimeWnd_Sp,
                                                                                         NormalPartRatio, LowPercent,
                                                                                         HighPercent, MinRiskIta,
                                                                                         MinRiskQ, 1)

        ## 安全参数可视化
        if sum(RiskMark) == 0:
            print("%s 无异常" % filename)
        try:
            z, R1, bc = GetSampleDataReady(r1, V, Fs, Sp, RiskMark, Ita1, Q, dt, p1, Soc, filename)
        except Exception as e:
            print(e)
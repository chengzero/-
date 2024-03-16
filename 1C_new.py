'''
傅里叶变换
2024年3月6日21:07:12
'''

import pandas as pd
import numpy as np
import os
import sqlite3
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是一种常用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import datetime

os.chdir(r'D:\code\data')


time1=datetime.datetime.now()


# # excel_name=r'E:\data\1c数据\2023.8.23柳南下行\2023.8.23柳南下行波形图对比分析记录表.xlsx'
# excel_name=r'E:\data\1c数据\2023.3.20柳南下行\2023-3-20 柳南下行波形图对比分析记录表.xlsx'
#
#
# def excel_read(excel_name):
#     df=pd.read_excel(excel_name)
#     df=df[['导高','公里标','V形']]
#     return df
#
# df_excel=excel_read(excel_name)
#
# # sql_name='2023-08-23柳南客专下行南宁局8车在前6号弓.db'
# sql_name='2023-03-20柳南客专下行南宁局1车在前6号弓.db'
#
# def sql_read(sql_name):
#
#     conn = sqlite3.connect(sql_name)
#     query = "SELECT name FROM sqlite_master WHERE type='table';"
#     tables = pd.read_sql_query(query, conn)
#     jc_tables = tables[tables['name'].str.startswith('WaveData')]
#     jc_tables=jc_tables['name'].values[0]
#     query = f"SELECT * FROM {jc_tables}"
#     df = pd.read_sql_query(query, conn)
#     conn.close()
#     df=df[['DIST', 'HEIGHT1','OPTSTAGGER1']]
#     df.columns = ['公里标', '导高', '拉出值']
#     #数据基本清洗
#     #删除导高为0的值
#     df=df[df['导高']!=0]
#     #删除公里标不对的行
#     df['公里标插值']=df['公里标']-df['公里标'].shift(-1)
#     df['公里标1']=df['公里标插值'].apply(lambda x:1 if abs(x)>50 else np.nan)
#     df['公里标1']=df['公里标1'].fillna(method='bfill')
#     df=df[df['公里标1']!=1]
#     df=df.reset_index(drop=True)
#
#     #筛选一张图像
#     df=df[df['公里标']>=212-1]
#     df = df[df['公里标'] <= 212+1]
#     return df
#
# df=sql_read(sql_name)
#
#
# # df.to_csv('df_clear.csv', index=False)
# # df=pd.read_csv('df_clear.csv')
#
# def sql_clear(df):
#     #绘图比较
#     df=df.reset_index(drop=True)
#     df['公里标1']=df['公里标']
#     df['公里标']=df.index
#     df['锚段']=df['拉出值']-df['拉出值'].shift(1)
#     df['锚段']=df['锚段'].apply(lambda x:1 if abs(x)>=150 else 0)
#
#
#     def hui_tu(df,str1):
#         plt.figure(figsize=(14, 4))
#         plt.subplot(311)
#         plt.plot(df[str1], df['导高'])
#         plt.title('导高')
#         plt.legend()
#
#         plt.subplot(312)
#         plt.plot(df[str1], df['拉出值'])
#         plt.title('拉出值')
#         plt.legend()
#         plt.tight_layout()
#
#         plt.subplot(313)
#         plt.plot(df[str1], df['锚段'])
#         plt.title('锚段')
#         plt.legend()
#         plt.tight_layout()
#         plt.show()
#
#     hui_tu(df, str1='公里标1')
#     hui_tu(df, str1='公里标')
#     print('此处观察一下波长的周期','1000个点有五个波峰，200个点一个周期')
#     return df
#
# df=sql_clear(df)
#
#
# # 因为后面分析都是针对拉出值的，所以此处将导高变量与拉出值进行交换
# series1=df['拉出值']
# df['拉出值']=df['导高']
# df['导高']=series1
#
#
# def avg_move(df):
#     df['拉出值1']=df['拉出值'].rolling(window=5, min_periods=1).mean()
#     plt.figure(figsize=(14, 4))
#     plt.subplot(211)
#     plt.plot(df['公里标'],df['拉出值'],color='blue')
#     plt.title('原始')
#     plt.legend()
#
#     plt.subplot(212)
#     plt.plot(df['公里标'], df['拉出值1'], color='red')
#     plt.title('平滑')
#     plt.legend()
#     plt.tight_layout()
#
#     plt.show()
#
#     df['拉出值']=df['拉出值1']
#     return df
#
# df=avg_move(df)

df=pd.read_csv('df.csv')
df=df.iloc[:,1:]

def fft_trans(df):

    df['拉出值']=df['拉出值']-5500
    def hui_tu(df,str1):
        plt.figure(figsize=(14, 4))
        plt.subplot(311)
        plt.plot(df[str1], df['导高'])
        plt.title('导高')
        plt.legend()

        plt.subplot(312)
        plt.plot(df[str1], df['拉出值'])
        plt.title('拉出值')
        plt.legend()
        plt.tight_layout()

        plt.subplot(313)
        plt.plot(df[str1], df['锚段'])
        plt.title('锚段')
        plt.legend()
        plt.tight_layout()
        plt.show()

    hui_tu(df, str1='公里标1')
    hui_tu(df, str1='公里标')

    des1=df.describe()

    # 应用快速傅里叶变换(FFT)
    fft_values = np.fft.fft(df['拉出值'])

    # 获取频率
    frequencies = np.fft.fftfreq(len(fft_values))

    # 绘制原始频谱图
    plt.figure(figsize=(14, 4))
    plt.plot(frequencies, np.abs(fft_values))
    plt.title('Original Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

    # 降噪: 由于这是对称的，我们只设置一边的频率阈值
    # 通常，如果没有噪声特定频率的先验知识，这个阈值需要根据情况尝试设置
    threshold = np.max(np.abs(fft_values)) * 0.1  # 示例阈值: 最大幅度的10%
    # threshold=0.5
    fft_values[np.abs(fft_values) < threshold] = 0

    # 绘制降噪后的频谱图
    plt.figure(figsize=(14, 4))
    plt.plot(frequencies, np.abs(fft_values))
    plt.title('Filtered Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

    # 应用逆傅里叶变换重建信号
    filtered_values = np.fft.ifft(fft_values)

    # 绘制原始信号与降噪后的信号对比图
    plt.figure(figsize=(14, 4))
    plt.plot(df['公里标'], df['拉出值'], label='Original', color='blue', linestyle='-')
    plt.plot(df['公里标'], filtered_values.real, label='Filtered', color='red', linestyle='-')
    plt.title('Original vs. Filtered Signal')
    plt.xlabel('Kilometer Mark')
    plt.ylabel('Pullout Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(14, 4))
    plt.subplot(211)
    plt.plot(df['公里标'], df['拉出值'],color='blue')
    plt.title('拉出值')
    plt.legend()

    plt.subplot(212)
    plt.plot(df['公里标'], filtered_values.real,color='red')
    plt.title('拉出值平滑')
    plt.legend()
    plt.tight_layout()
    plt.show()


    df['拉出值降噪'] = filtered_values.real
    df['拉出值']=df['拉出值降噪']
    return df

df=fft_trans(df)


def peak_troughs(df):


    y=df['拉出值']
    plt.figure(figsize=(14,4))
    plt.plot(y)
    plt.show()
    # 寻找波峰
    peaks, _ = find_peaks(y, width=40,distance=40)
    # 寻找波谷
    troughs, _ = find_peaks(-y, width=40,distance=40)
    df['波峰波谷'] = 0
    df.loc[peaks, '波峰波谷'] = 1
    df.loc[troughs, '波峰波谷'] = -1

    plt.figure(figsize=(14, 4))
    plt.subplot(311)
    plt.plot(df.index, df['拉出值'], color='blue')
    plt.title('拉出值')
    plt.legend()

    plt.subplot(312)
    plt.plot(df.index, df['波峰波谷'], color='red')
    plt.title('峰谷')
    plt.legend()
    plt.tight_layout()

    plt.subplot(313)
    plt.plot(df.index, df['导高'], color='red')
    plt.title('导高')
    plt.legend()
    plt.tight_layout()

    plt.show()


    # 将波峰的索引与波谷的索引作为list拼接到一起
    peaks=peaks.tolist()
    troughs=troughs.tolist()
    # peaks.extend(troughs)
    # peaks.sort()
    # extrema=peaks
    segments_correct = {}

    #将锚段前后的extrema剔除
    extrema1=troughs
    for i in range(len(extrema1) - 1):
        # i, i+1, i+2 形成了波峰-波谷-波峰或波谷-波峰-波谷
        segment_data_correct = df.iloc[extrema1[i]:extrema1[i + 1]]
        plt.plot( segment_data_correct['拉出值'])
        plt.show()
        shang=segment_data_correct['公里标1'].max()
        xia=segment_data_correct['公里标1'].min()
        if segment_data_correct[segment_data_correct['锚段']==1].empty:
            segments_correct[f'{xia}-{shang}'] = segment_data_correct
        else:
            print('此处有锚段')
    print('运行完成')

    return segments_correct



segments_correct=peak_troughs(df)



def feature_engineering(segments_correct,df_excel):
    df_empty=pd.DataFrame()
    list_excel=df_excel['公里标'].tolist()
    for i in tqdm(segments_correct.keys()):
        # i= list(segments_correct.keys())[0]
        data1=segments_correct[i]
        list_sql=data1['公里标1'].tolist()
        list_x=[x for x in list_excel if x in list_sql]
        data2=data1[data1['波峰波谷']!=0]
        # index1=data2['波峰波谷'].tolist()[0]
        index2 = data2['波峰波谷'].tolist()[1]
        # index3 = data2['波峰波谷'].tolist()[2]
        index_2=data2[data2['波峰波谷']==index2]['公里标1'].values[0]

        if len(list_x)>0 :
            que_xian=1
            feng_gu=index_2
            excel_feng_gu = list_x[0]
            v_excel=df_excel[df_excel['公里标']==list_x[0]]['V形'].values[0]
        else:
            que_xian=0
            feng_gu=np.nan
            excel_feng_gu=np.nan
            v_excel=np.nan
        zhou_qi=data1.shape[0]
        zhen_fu=data1['拉出值'].max()-data1['拉出值'].min()


        skewness = data1['拉出值'].skew()  #偏度>0，右偏，右边拖尾
        kurtosis = data1['拉出值'].kurt()  #峰度<0，平坦
        std1=data1['拉出值'].std()
        if data2['波峰波谷'].tolist()[0]==-1 and data2['波峰波谷'].tolist()[1]==1\
                and data2['波峰波谷'].tolist()[2]==-1:
            V='倒V'
            dao_gao=data2[data2['波峰波谷']==1]['拉出值'].values[0]
        elif data2['波峰波谷'].tolist()[0]==1 and data2['波峰波谷'].tolist()[1]==-1\
                and data2['波峰波谷'].tolist()[2]==1:
            V='V'
            dao_gao = data2[data2['波峰波谷'] == -1]['拉出值'].values[0]
        else:
            V='其他'



        data2=pd.DataFrame({'km':[i],'zhou_qi':[zhou_qi],'zhen_fu':[zhen_fu], \
                            'skewness':[skewness],'kurtosis':[kurtosis],'std1':[std1],
                            'que_xian':[que_xian],'feng_gu':[feng_gu],
                            'excel_feng_gu':[excel_feng_gu],'V':[V],'V_excel':[v_excel]})
        df_empty=pd.concat([df_empty,data2],axis=0)

    return df_empty

df_empty=feature_engineering(segments_correct,df_excel)




def que_xian_correct(df_empty):
    df_empty['que_xian1'] = df_empty.apply(
        lambda x: 1 if x['que_xian'] == 1 and x['V'] == x['V_excel'] else 0, axis=1)
    return df_empty

df_empty=que_xian_correct(df_empty)


sql_name=sql_name.split('.')[0]
df_empty.to_csv(f'{sql_name}.csv')

time2=datetime.datetime.now()

print('运行时间',time2-time1)




"""
原始数据清洗
1、读取csv文件
2、字段数据信息处理
    1、蓄电池组1与蓄电池组2进行区分，将数据拆分堆叠
    2、判断车辆上电断电时刻
    3、只取vcb断开->vcb闭合期间的数据
    4、温度取两个变量的平均值
    5、记录放电时长
    6、当电压数据为0时，判断车辆数据是否是 上电时刻或断开充电机空开 状态，如果是，抛弃该行数据
    7、vcb断开后两分钟数据抛弃
    8、vcb闭合后10秒 数据抛弃
3、数据另存
    将处理数据存储为txt格式文件，三列数据，分别为，电池组平均温度、电池组电压、放电时长

"""
import datetime
import os

import pandas as pd
import numpy as np
from tqdm import tqdm

# 数据文件的字段
columns = [
    'train_type', 'train_id', 'coach_no', 'time_value', 'speed', 'trainsite_id', 'vcb_status',
    'charge_battery_group1_tem1', 'charge_battery_group1_tem2', 'charge_battery_group2_tem1',
    'charge_battery_group2_tem2',
    'charger_out_v1', 'charger_out_v2', 'UB_air_brake_effective', 'embrake_ub_loop_dis',
    'network_volt', 'mcr_signal1', 'mcr_signal2', 'union_signal'
]

# cn = [
#     车型 , 车号,车厢,时间,列车速度 ,车次,	VCB状态
#     蓄电池1温度1	蓄电池1温度2	蓄电池2温度1	蓄电池2温度2
#     充电机1输出电压（既蓄电池1电压）	充电机2输出电压（既蓄电池2电压）	UB空气制动有效	紧急制动UB环路状态
#     网压	主控信号1,主控信号	列车头尾车联挂信号
# ]

# 保留字段
keep_columns = [
    'train_type', 'train_id',
    'coach_no',  # 车厢编号
    'vcb_status',  # VCB状态
    'time_value',  # 时间数据，格式为 20210101000000 YMDHMS
    'charge_battery_group1_tem1', 'charge_battery_group1_tem2',
    'charge_battery_group2_tem1', 'charge_battery_group2_tem2',
    'charge_battery_group_tem_1',  # 蓄电池组温度(平均)
    'charge_battery_group_tem_2',  # 蓄电池组温度(平均)
    'charger_out_v1',  # 电压
    'charger_out_v2',  # 电压
    'discharge_duration'  # 放电时长
]

charge_group_1_columns = [
    'train_type', 'train_id',
    'coach_no',  # 车厢编号
    'vcb_status',  # VCB状态
    'time_value',  # 时间数据，格式为 20210101000000 YMDHMS
    'charge_battery_group1_tem1', 'charge_battery_group1_tem2',
    # 'charge_battery_group_tem_1',  # 蓄电池组温度(平均)
    'charger_out_v1',  # 电压
    # 'discharge_duration'  # 放电时长
]

charge_group_2_columns = [
    'train_type', 'train_id',
    'coach_no',  # 车厢编号
    'vcb_status',  # VCB状态
    'time_value',  # 时间数据，格式为 20210101000000 YMDHMS
    'charge_battery_group2_tem1', 'charge_battery_group2_tem2',
    # 'charge_battery_group_tem_2',  # 蓄电池组温度(平均)
    'charger_out_v2',  # 电压
    # 'discharge_duration'  # 放电时长
]

charge_train_columns_1 = {
    'time_value': 'time_value',

    # 'vcb_status': 'vcb_status',  # VCB状态
    # 'charge_battery_group_tem_1': 'charge_battery_tem',  # 蓄电池组温度(平均)
    # 'charger_out_v1': 'charger_out',  # 电压
    # 'discharge_duration': 'discharge_duration'  # 放电时长
}

charge_train_columns_2 = {
    'time_value': 'time_value',

    # 'vcb_status': 'vcb_status',  # VCB状态
    # 'charge_battery_group_tem_2': 'charge_battery_tem',  # 蓄电池组温度(平均)
    # 'charger_out_v2': 'charger_out',  # 电压
    # 'discharge_duration': 'discharge_duration'  # 放电时长
}


def get_vcb(group, coach_no):
    s = group.loc[group['coach_no'] == coach_no, 'vcb_status']
    if s.empty:
        return None
    return s.iloc[0]


def read_csv(file_path):
    """
    读取csv文件
    :param file_path:
    :return:
    """

    df = pd.read_csv(file_path)

    df['coach_no'] = df['coach_no'].astype(int)

    result_df = df.copy()

    # 只保留 coach_no=01的 数据
    result_df = result_df.loc[result_df['coach_no'] == 1]
    return result_df



    # # 电池组1、2
    # final_cols_1 = [c for c in charge_group_1_columns if c in result_df.columns]
    # final_cols_2 = [c for c in charge_group_2_columns if c in result_df.columns]
    #
    # result_df_1 = result_df[final_cols_1]
    # result_df_2 = result_df[final_cols_2]
    #
    # # 对两组数据根据time_value进行去重
    # result_df_1 = result_df_1.drop_duplicates(subset=['time_value'])
    # result_df_2 = result_df_2.drop_duplicates(subset=['time_value'])
    #
    # return result_df_1, result_df_2


def read_and_save_as_csv(file_path, save_path_1, save_path_2):
    try:
        # df_1, df_2 = read_csv(file_path)
        # # df.to_csv(save_path, index=False)
        # df_1.to_csv(save_path_1, index=False)
        # df_2.to_csv(save_path_2, index=False)
        df = read_csv(file_path)
        df.to_csv(save_path_1, index=False)

        print(f"数据保存成功：{save_path_1} and {save_path_2}")
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()


def csv_to_txt(file_path, clean_txt_dir_path):
    try:
        df = pd.read_csv(file_path)
        # 原有的列
        final_cols_1 = [c for c in charge_train_columns_1 if c in df.columns]
        # final_cols_2 = [c for c in charge_train_columns_2 if c in df.columns]

        df = df[final_cols_1]
        df.rename(columns=charge_train_columns_1, inplace=True)

        file_name = file_path.split("\\")[-1]
        path_ = os.path.join(clean_txt_dir_path, file_name[:-4] + ".txt")
        df.to_csv(path_, index=False, header=False)
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':

    today_ = datetime.datetime.now().strftime("%Y_%m_%d")

    dir_path = r"D:\Work\LLM\GitHub\Large-Time-Series-Model\csv_data\original_data"
    clean_dir_path = r"D:\Work\LLM\GitHub\Large-Time-Series-Model\csv_data\clean_data_new"
    clean_txt_dir_path = r"D:\Work\LLM\GitHub\Large-Time-Series-Model\dataset\xudianchi\train\111"

    if os.path.exists(clean_dir_path) == False:
        os.makedirs(clean_dir_path)
    if os.path.exists(clean_txt_dir_path) == False:
        os.makedirs(clean_txt_dir_path)

    # 将数据进行整理清洗，并保存为 csv 文件
    for file in tqdm(os.listdir(dir_path)):
        file_path = os.path.join(dir_path, file)
        file_name, file_type = os.path.splitext(file)
        save_path_1 = os.path.join(clean_dir_path, f"{file_name}_group_1.csv")
        save_path_2 = os.path.join(clean_dir_path, f"{file_name}_group_2.csv")
        read_and_save_as_csv(file_path, save_path_1, save_path_2)

    # # 将 csv 文件转换为 txt 文件
    for file in tqdm(os.listdir(clean_dir_path)):
        file_path = os.path.join(clean_dir_path, file)
        csv_to_txt(file_path, clean_txt_dir_path)

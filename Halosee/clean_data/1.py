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
    'vcb_status',  # VCB状态
    'charge_battery_group1_tem1', 'charge_battery_group1_tem2',
    'charge_battery_group2_tem1', 'charge_battery_group2_tem2',
    'charger_out_v1',  # 电压
    'charger_out_v2',  # 电压

]

charge_group_1_columns = [
    'train_type', 'train_id',
    'coach_no',  # 车厢编号
    'vcb_status',  # VCB状态
    'time_value',  # 时间数据，格式为 20210101000000 YMDHMS
    'charge_battery_group1_tem1',
    'charge_battery_group1_tem2',
    'charge_battery_group_tem_1',  # 蓄电池组温度(平均)
    'charger_out_v1',  # 电压
    'discharge_duration'  # 放电时长
]

charge_group_2_columns = [
    'train_type', 'train_id',
    'coach_no',  # 车厢编号
    'vcb_status',  # VCB状态
    'time_value',  # 时间数据，格式为 20210101000000 YMDHMS
    'charge_battery_group2_tem1', 'charge_battery_group2_tem2',
    'charge_battery_group_tem_2',  # 蓄电池组温度(平均)
    'charger_out_v2',  # 电压
    'discharge_duration'  # 放电时长
]

charge_train_columns_1 = {
    'time_value': 'time_value',

    'vcb_status': 'vcb_status',  # VCB状态
    'charge_battery_group_tem_1': 'charge_battery_tem',  # 蓄电池组温度(平均)
    'charger_out_v1': 'charger_out',  # 电压
    'discharge_duration': 'discharge_duration'  # 放电时长
}

charge_train_columns_2 = {
    'time_value': 'time_value',
    'vcb_status': 'vcb_status',  # VCB状态
    'charge_battery_group_tem_2': 'charge_battery_tem',  # 蓄电池组温度(平均)
    'charger_out_v2': 'charger_out',  # 电压
    'discharge_duration': 'discharge_duration'  # 放电时长
}



def read_csv(file_path):
    """
    读取csv文件
    :param file_path:
    :return:
    """

    print(f" >>> 开始读取文件: {file_path}...")
    df = pd.read_csv(file_path)

    # 确保 coach_no 是整数类型，以便进行数字比较
    # 如果源数据是 "01", "02" 这种字符串，这步会将其转为 1, 2
    df['coach_no'] = df['coach_no'].astype(int)
    df['charge_battery_group1_tem1'] = df['charge_battery_group1_tem1'].astype(float)
    df['charge_battery_group1_tem2'] = df['charge_battery_group1_tem2'].astype(float)
    df['charge_battery_group2_tem1'] = df['charge_battery_group2_tem1'].astype(float)
    df['charge_battery_group2_tem2'] = df['charge_battery_group2_tem2'].astype(float)
    df['charger_out_v1'] = df['charger_out_v1'].astype(float)
    df['charger_out_v2'] = df['charger_out_v2'].astype(float)


    print(" >>> 开始填充1~4车厢以及5~8车厢的空值...")
    # 1. 创建编组标识 (Group ID)
    # 使用 pd.cut 可以非常方便地划分区间，同时也容易扩展到16编组
    # bins=[0, 4, 8] 表示 (0,4] 为第一组, (4,8] 为第二组
    # labels=[1, 2] 对应组名
    df['group_id'] = pd.cut(df['coach_no'], bins=[0, 4, 8, 12, 16], labels=[1, 2, 3, 4])
    # 2. 定义聚合逻辑
    # 我们需要在同时间、同编组内找到那个“有值的行”。
    # 这里的逻辑是：取组内的最大值(max)或第一个非空值。
    # 通常状态码用 max 比较方便 (例如 1 或 2 会覆盖 NaN)
    # transform('max') 会计算出每组的最大值，并将该值赋给组内的每一行，保持行数不变
    filled_vcb = df.groupby(['time_value', 'group_id'], observed=True)['vcb_status'].transform('max')
    temp_1_1 = df.groupby(['time_value', 'group_id'], observed=True)['charge_battery_group1_tem1'].transform('max')
    temp_1_2 = df.groupby(['time_value', 'group_id'], observed=True)['charge_battery_group1_tem2'].transform('max')
    temp_2_1 = df.groupby(['time_value', 'group_id'], observed=True)['charge_battery_group2_tem1'].transform('max')
    temp_2_2 = df.groupby(['time_value', 'group_id'], observed=True)['charge_battery_group2_tem2'].transform('max')
    charger_out_v1 = df.groupby(['time_value', 'group_id'], observed=True)['charger_out_v1'].transform('max')
    charger_out_v2 = df.groupby(['time_value', 'group_id'], observed=True)['charger_out_v2'].transform('max')

    # 3. 更新 vcb_status
    # 将计算出的广播值赋回原列，同时将原本就是 NaN 且没广播到值的（即整组都为空）填为 0
    df['vcb_status'] = filled_vcb.fillna(0)
    df['charge_battery_group1_tem1'] = temp_1_1.fillna(np.nan)
    df['charge_battery_group1_tem2'] = temp_1_2.fillna(np.nan)
    df['charge_battery_group2_tem1'] = temp_2_1.fillna(np.nan)
    df['charge_battery_group2_tem2'] = temp_2_2.fillna(np.nan)
    df['charger_out_v1'] = charger_out_v1.fillna(np.nan)
    df['charger_out_v2'] = charger_out_v2.fillna(np.nan)
    # 4. 删除辅助列
    df.drop(columns=['group_id'], inplace=True)

    df = df.fillna(np.nan)

    # return df

    print(" >>> 开始填充即整组车厢都为空的值【如果分组内有vcb一半为0，另一半不为0的数据，则将为0的数据填充为不为0的数据】...")
    # # 判断，如果分组内有vcb一半为0，另一半不为0的数据，则将为0的数据填充为不为0的数据
    # group_data = df.groupby('time_value')
    # for time_value, df_group in group_data:
    #     # 获取分组内所有的vcb状态
    #     vcb_status = df_group['vcb_status'].values
    #     # 将vcb_statu为0的数据填充成vcb_statu不为0的数据
    #     # 找到组内非 0 的有效值
    #     non_zero_values = vcb_status[vcb_status != 0]
    #
    #     # 若存在有效值，则用第一个非 0 值填充 0
    #     if len(non_zero_values) > 0:
    #         fill_value = non_zero_values[0]
    #         df.loc[df_group.index, 'vcb_status'] = df_group['vcb_status'].replace(0, fill_value)
    #
    # print(" >>> 填充完成...")

    # 找到每个 time_value 下第一个非 0 的 vcb_status
    # fill_map = (
    #     df['vcb_status']
    #     .where(df['vcb_status'] != 0)
    #     .groupby(df['time_value'])
    #     .transform('first')
    # )
    #
    # # 只填充 vcb_status == 0 的位置
    # df['vcb_status'] = df['vcb_status'].where(
    #     df['vcb_status'] != 0,
    #     fill_map
    # )

    for col in keep_columns:
        # 定义缺失规则
        is_missing = (df[col] == 0) if col == 'vcb_status' else df[col].isna()

        # 在每个 time_value 内，提取“唯一有效值”
        def unique_valid_value(x):
            vals = x[~is_missing.loc[x.index]]
            uniq = vals.unique()
            return uniq[0] if len(uniq) == 1 else np.nan

        fill_map = (
            df.groupby('time_value')[col]
            .apply(unique_valid_value)
        )

        # 仅在缺失处填充
        df.loc[is_missing, col] = df.loc[is_missing, 'time_value'].map(fill_map)

    print(">>> 对称补全完成")
    return  df
    result_df = df.copy()
    result_df = result_df.fillna(np.nan)
    #
    #
    # def fill_half_group_by_time(result_df):
    #
    #     # 需要做“组内互补填充”的字段
    #     fill_cols = [
    #         'charge_battery_group1_tem1',
    #         'charge_battery_group1_tem2',
    #         'charge_battery_group2_tem1',
    #         'charge_battery_group2_tem2',
    #         'charger_out_v1',
    #         'charger_out_v2',
    #     ]
    #
    #     for time_value, g in result_df.groupby('time_value'):
    #         idx_1_4 = g[g['coach_no'].isin([1, 2, 3, 4])].index
    #         idx_5_8 = g[g['coach_no'].isin([5, 6, 7, 8])].index
    #
    #         for col in fill_cols:
    #             v_1_4 = result_df.loc[idx_1_4, col]
    #             v_5_8 = result_df.loc[idx_5_8, col]
    #
    #             has_1_4 = v_1_4.notna().any()
    #             has_5_8 = v_5_8.notna().any()
    #
    #             # 1~4 有值，5~8 全空 → 用 1~4 填 5~8
    #             if has_1_4 and not has_5_8:
    #                 fill_value = v_1_4.dropna().iloc[0]
    #                 result_df.loc[idx_5_8, col] = fill_value
    #
    #             # 5~8 有值，1~4 全空 → 用 5~8 填 1~4
    #             elif has_5_8 and not has_1_4:
    #                 fill_value = v_5_8.dropna().iloc[0]
    #                 result_df.loc[idx_1_4, col] = fill_value
    #
    #     return result_df
    #
    # result_df = fill_half_group_by_time(result_df)

    # 添加列 charge_battery_group_tem_1 = 蓄电池组1平均温度， charge_battery_group_tem_2 = 蓄电池组2平均温度
    print(" >>> 添加列 蓄电池组1平均温度，蓄电池组2平均温度...")
    # result_df['charge_battery_group_tem_1'] = result_df.apply(
    #     lambda x: (x['charge_battery_group1_tem1'] + x['charge_battery_group1_tem2']) / 2, axis=1)
    # result_df['charge_battery_group_tem_2'] = result_df.apply(
    #     lambda x: (x['charge_battery_group2_tem1'] + x['charge_battery_group2_tem2']) / 2, axis=1)
    #
    # # 先按时间从小到达排序，再按车厢排序，然后将车厢按顺序堆叠
    # print("执行排序：时间 -> 车厢...")
    # # 将时间转换为字符串或直接利用数值排序，转为Datetime方便后续计算
    # # time_value 格式为 YYYYMMDDHHMMSS 的数字或字符串
    # result_df = result_df.sort_values(by=['coach_no', 'time_value'], ascending=[True, True])

    # print("正在清洗温度和电压数据...")
    #
    # # 清洗温度
    # temp_cols = ['charge_battery_group1_tem1', 'charge_battery_group1_tem2', 'charge_battery_group2_tem1',
    #              'charge_battery_group2_tem2']
    # result_df = result_df.dropna(subset=temp_cols, how='all')
    # result_df = result_df[~result_df[temp_cols].astype(str).apply(lambda x: x.str.startswith('-').all(), axis=1)]
    #
    # # 清洗电压
    # volt_cols = ['charger_out_v1', 'charger_out_v2']
    # result_df = result_df.dropna(subset=volt_cols, how='all')
    # result_df = result_df[~result_df[volt_cols].astype(str).apply(lambda x: x.eq('0').all(), axis=1)]


    # 计算平均温度
    result_df['charge_battery_group_tem_1'] = result_df[['charge_battery_group1_tem1', 'charge_battery_group1_tem2']].mean(axis=1)
    result_df['charge_battery_group_tem_2'] = result_df[['charge_battery_group2_tem1', 'charge_battery_group2_tem2']].mean(axis=1)

    # 最终排序：车厢 -> 时间
    print(" >>> 执行最终排序...")
    result_df = result_df.sort_values(by=['coach_no', 'time_value'], ascending=[True, True])

    print(" >>> 计算放电时长...")
    # 添加字段discharge_duration

    # 从第一行开始，铆钉字段time_value，第一行为1，接着往下+1，当time_value不连续时，则认为该段结束，不连续的下一行为下一段数据，从1开始
    # 初始化 discharge_duration
    discharge_duration = [1]

    # 获取result_df的索引列表
    indices = result_df.index.tolist()

    # 逐行计算
    for i in range(1, len(result_df)):
        # 使用实际的索引值来访问数据
        prev_idx = indices[i - 1]
        curr_idx = indices[i]

        prev_time = result_df.loc[prev_idx, "time_value"]
        curr_time = result_df.loc[curr_idx, "time_value"]

        # 将时间戳转换为datetime格式进行比较
        prev_time_dt = pd.to_datetime(str(prev_time), format='%Y%m%d%H%M%S')
        curr_time_dt = pd.to_datetime(str(curr_time), format='%Y%m%d%H%M%S')

        # 判断时间差是否为1秒
        if (curr_time_dt - prev_time_dt).total_seconds() == 1:
            discharge_duration.append(discharge_duration[-1] + 1)
        else:
            discharge_duration.append(0)

    # 写入新列
    result_df["discharge_duration"] = discharge_duration

    # 电池组1、2
    final_cols_1 = [c for c in charge_group_1_columns if c in result_df.columns]
    final_cols_2 = [c for c in charge_group_2_columns if c in result_df.columns]

    result_df_1 = result_df[final_cols_1]
    result_df_2 = result_df[final_cols_2]

    result_df_1_coach1 = result_df_1[result_df_1['coach_no'] == 1]
    result_df_1_coach8 = result_df_1[result_df_1['coach_no'] == 8]

    result_df_2_coach1 = result_df_2[result_df_2['coach_no'] == 1]
    result_df_2_coach8 = result_df_2[result_df_2['coach_no'] == 8]


    return result_df_1_coach1, result_df_1_coach8, result_df_2_coach1, result_df_2_coach8


def read_and_save_as_csv(file_path, save_path_1, save_path_2, save_path_3, save_path_4):
    try:
        # df_1, df_2 = read_csv(file_path)
        # df_1.to_csv(save_path_1, index=False)
        # df_2.to_csv(save_path_2, index=False)
        result_df_1_coach1= read_csv(file_path)
        result_df_1_coach1.to_csv(save_path_1, index=False)
        # result_df_1_coach8.to_csv(save_path_2, index=False)
        # result_df_2_coach1.to_csv(save_path_3, index=False)
        # result_df_2_coach8.to_csv(save_path_4, index=False)


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
        final_cols_2 = [c for c in charge_train_columns_2 if c in df.columns]
        # 新的列
        if len(final_cols_1) > 3:
            print("正在处理1：", final_cols_1)
            df = df[final_cols_1]
            df.rename(columns=charge_train_columns_1, inplace=True)
        if len(final_cols_2) > 3:
            print("正在处理2：", final_cols_2)
            df = df[final_cols_2]
            df.rename(columns=charge_train_columns_2, inplace=True)
        file_name = file_path.split("\\")[-1]
        path_ = os.path.join(clean_txt_dir_path, file_name[:-4] + ".txt")
        df.to_csv(path_, index=False, header=False)
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':

    today_ = datetime.datetime.now().strftime("%Y_%m_%d")

    dir_path = r"D:\Work\LLM\GitHub\Large-Time-Series-Model\csv_data\1_original_data"
    clean_dir_path = r"D:\Work\LLM\GitHub\Large-Time-Series-Model\csv_data\2_clean_data_new"
    clean_txt_dir_path = r"D:\Work\LLM\GitHub\Large-Time-Series-Model\csv_data\3_txt_data"

    if os.path.exists(clean_dir_path) == False:
        os.makedirs(clean_dir_path)
    if os.path.exists(clean_txt_dir_path) == False:
        os.makedirs(clean_txt_dir_path)

    # 将数据进行整理清洗，并保存为 csv 文件
    for file in tqdm(os.listdir(dir_path)):
        file_path = os.path.join(dir_path, file)
        file_name, file_type = os.path.splitext(file)
        save_path_1 = os.path.join(clean_dir_path, f"{file_name}_group_1_1.csv")
        save_path_2 = os.path.join(clean_dir_path, f"{file_name}_group_1_8.csv")
        save_path_3 = os.path.join(clean_dir_path, f"{file_name}_group_2_1.csv")
        save_path_4 = os.path.join(clean_dir_path, f"{file_name}_group_2_8.csv")
        read_and_save_as_csv(file_path, save_path_1, save_path_2, save_path_3, save_path_4)
        break

    # # # 将 csv 文件转换为 txt 文件
    # for file in tqdm(os.listdir(clean_dir_path)):
    #     file_path = os.path.join(clean_dir_path, file)
    #     csv_to_txt(file_path, clean_txt_dir_path)

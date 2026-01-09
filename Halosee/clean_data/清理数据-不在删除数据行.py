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
    # 'charge_battery_group_tem_1',  # 蓄电池组温度(平均)
    # 'charge_battery_group_tem_2',  # 蓄电池组温度(平均)
    'charger_out_v1',  # 电压
    'charger_out_v2',  # 电压
    # 'discharge_duration'  # 放电时长
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
    # 'discharge_duration'  # 放电时长
]

charge_group_2_columns = [
    'train_type', 'train_id',
    'coach_no',  # 车厢编号
    'vcb_status',  # VCB状态
    'time_value',  # 时间数据，格式为 20210101000000 YMDHMS
    'charge_battery_group2_tem1', 'charge_battery_group2_tem2',
    'charge_battery_group_tem_2',  # 蓄电池组温度(平均)
    'charger_out_v2',  # 电压
    # 'discharge_duration'  # 放电时长
]

charge_train_columns_1 = {
    'time_value': 'time_value',

    'vcb_status': 'vcb_status',  # VCB状态
    'charge_battery_group_tem_1': 'charge_battery_tem',  # 蓄电池组温度(平均)
    'charger_out_v1': 'charger_out',  # 电压
    # 'discharge_duration': 'discharge_duration'  # 放电时长
}

charge_train_columns_2 = {
    'time_value': 'time_value',
    'vcb_status': 'vcb_status',  # VCB状态
    'charge_battery_group_tem_2': 'charge_battery_tem',  # 蓄电池组温度(平均)
    'charger_out_v2': 'charger_out',  # 电压
    # 'discharge_duration': 'discharge_duration'  # 放电时长
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


    print(" >>> 开始填充即整组车厢都为空的值【如果分组内有vcb一半为0，另一半不为0的数据，则将为0的数据填充为不为0的数据】...")

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
    result_df = df.copy()
    result_df = result_df.fillna(np.nan)

    # 添加列 charge_battery_group_tem_1 = 蓄电池组1平均温度， charge_battery_group_tem_2 = 蓄电池组2平均温度
    print(" >>> 添加列 蓄电池组1平均温度，蓄电池组2平均温度...")

    # 计算平均温度
    result_df['charge_battery_group_tem_1'] = result_df[['charge_battery_group1_tem1', 'charge_battery_group1_tem2']].mean(axis=1)
    result_df['charge_battery_group_tem_2'] = result_df[['charge_battery_group2_tem1', 'charge_battery_group2_tem2']].mean(axis=1)

    # 最终排序：车厢 -> 时间
    print(" >>> 执行最终排序...")
    result_df = result_df.sort_values(by=['coach_no', 'time_value'], ascending=[True, True])


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
        result_df_1_coach1, result_df_1_coach8, result_df_2_coach1, result_df_2_coach8 = read_csv(file_path)
        result_df_1_coach1.to_csv(save_path_1, index=False)
        result_df_1_coach8.to_csv(save_path_2, index=False)
        result_df_2_coach1.to_csv(save_path_3, index=False)
        result_df_2_coach8.to_csv(save_path_4, index=False)

        print(f"数据保存成功：{save_path_1} and {save_path_2}")
    except Exception as e:
        print(f"处理失败1: {e}")
        import traceback
        traceback.print_exc()

def fix_voltage_jumps(df, vol_col_name, vcb_col_name, threshold=1.0):
    """
    修复 VCB 状态不变时的电压不规则跳变。
    逻辑：如果 Time_0 和 Time_3 电压一致（或接近），且 VCB 状态一致，
          则将 Time_1 和 Time_2 的电压修正为 Time_0 的值。
    """
    # 1. 提取数据为 Numpy 数组（计算速度比 DataFrame 快得多）
    # 注意：这里使用 copy() 防止直接修改原数据，除非你确定要 inplace
    vals = df[vol_col_name].values.copy()
    vcbs = df[vcb_col_name].values

    n = len(df)

    print(">>> 开始修正电压跳变...")

    # --- 情况 A: 修正跨度为 2 的跳变 (115, 114, 114, 115 -> 115, 115, 115, 115) ---
    # 我们看的是 i, i+1, i+2, i+3 这一组
    # 逻辑：Val[i] ≈ Val[i+3] 且 VCB[i]...VCB[i+3] 全相等

    # 构造错位视图
    v_t0 = vals[:-3]  # 0 ~ N-3
    v_t3 = vals[3:]  # 3 ~ N

    s_t0 = vcbs[:-3]
    s_t1 = vcbs[1:-2]
    s_t2 = vcbs[2:-1]
    s_t3 = vcbs[3:]

    # 判断电压是否“回归”了 (允许一点点误差，比如 0.5V，如果是整数可以用 ==)
    # 如果数据是严格整数，可以用 v_t0 == v_t3
    voltage_match = np.abs(v_t0 - v_t3) < threshold

    # 判断 VCB 在这段时间内是否完全没变
    vcb_stable = (s_t0 == s_t1) & (s_t1 == s_t2) & (s_t2 == s_t3)

    # 找到需要修正的索引掩码 (Mask)
    mask_gap2 = voltage_match & vcb_stable

    # 获取需要修正的行索引 (相对于原数组的 offset)
    # np.where 返回的是 tuple，取 [0]
    indices = np.where(mask_gap2)[0]

    if len(indices) > 0:
        print(f"检测到 {len(indices)} 处跨度为2的跳变，正在修正...")
        # 修正中间两个点 (i+1, i+2) 为 i 的值
        # 注意：这里我们用 t0 的值去覆盖 t1 和 t2
        vals[indices + 1] = vals[indices]
        vals[indices + 2] = vals[indices]

    # --- 情况 B: 修正跨度为 1 的跳变 (115, 114, 115 -> 115, 115, 115) ---
    # 逻辑同上，只是窗口变小，防止有些只有 1 个点的尖峰漏网

    v_t0_s = vals[:-2]
    v_t2_s = vals[2:]

    s_t0_s = vcbs[:-2]
    s_t1_s = vcbs[1:-1]
    s_t2_s = vcbs[2:]

    voltage_match_s = np.abs(v_t0_s - v_t2_s) < threshold
    vcb_stable_s = (s_t0_s == s_t1_s) & (s_t1_s == s_t2_s)

    mask_gap1 = voltage_match_s & vcb_stable_s
    indices_s = np.where(mask_gap1)[0]

    if len(indices_s) > 0:
        print(f"检测到 {len(indices_s)} 处跨度为1的跳变，正在修正...")
        vals[indices_s + 1] = vals[indices_s]

    # 将修正后的数组写回 DataFrame
    df[vol_col_name] = vals
    return df

def csv_to_txt(file_path, clean_txt_dir_path):
    try:
        print("正在处理：", file_path)
        df = pd.read_csv(file_path)
        print(f">>> 获取列, 长度为{len(df)}...")
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

        # 处理数据的单点抖动
        if 'charger_out' in df.columns and 'vcb_status' in df.columns:
            df = fix_voltage_jumps(df, 'charger_out', 'vcb_status')

        if not df.empty:
            # 确保数据是数值型，否则比较大小时会报错 (可选，根据你的数据情况)
            # df = df.apply(pd.to_numeric, errors='coerce')

            # 条件 1: 去除行中有负数的
            # 逻辑：保留那些 "所有值都大于等于0" 的行
            # (df >= 0).all(axis=1) 返回一个布尔序列，如果一行全是正数则为True
            df = df[(df >= 0).all(axis=1)]

            # 条件 2: 去除行中倒数第二列是 0 的
            # 逻辑：保留 "倒数第二列不等于0" 的行
            if df.shape[1] >= 2:  # 确保至少有两列
                df = df[df.iloc[:, -2] != 0]

            # 条件 3: 去除一行只有三个数值的
            # 这里的理解是：如果这一行只有3个有效值（非空值），则去除。
            # 逻辑：保留 "有效数值个数 > 3" 的行
            # count(axis=1) 会计算每一行非 NaN 的数量
            df = df[df.count(axis=1) > 3]

        print(f">>> 清洗后行数: {len(df)}")

        # 添加新列,放电时长
        print(" >>> 计算放电时长...")
        # 添加字段discharge_duration

        # 从第一行开始，铆钉字段time_value，第一行为1，接着往下+1，当time_value不连续时，则认为该段结束，不连续的下一行为下一段数据，从1开始
        # 初始化 discharge_duration
        discharge_duration = [0]

        # 获取result_df的索引列表
        indices = df.index.tolist()

        # 逐行计算
        for i in range(1, len(df)):
            # 使用实际的索引值来访问数据
            prev_idx = indices[i - 1]
            curr_idx = indices[i]

            prev_time = df.loc[prev_idx, "time_value"]
            curr_time = df.loc[curr_idx, "time_value"]

            # 将时间戳转换为datetime格式进行比较
            prev_time_dt = pd.to_datetime(str(prev_time), format='%Y%m%d%H%M%S')
            curr_time_dt = pd.to_datetime(str(curr_time), format='%Y%m%d%H%M%S')

            # 判断时间差是否为1秒
            if (curr_time_dt - prev_time_dt).total_seconds() == 1:
                discharge_duration.append(discharge_duration[-1] + 1)
            else:
                discharge_duration.append(0)

        # 写入新列
        df["discharge_duration"] = discharge_duration


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

    # # 将 csv 文件转换为 txt 文件
    for file in tqdm(os.listdir(clean_dir_path)):
        file_path = os.path.join(clean_dir_path, file)
        csv_to_txt(file_path, clean_txt_dir_path)

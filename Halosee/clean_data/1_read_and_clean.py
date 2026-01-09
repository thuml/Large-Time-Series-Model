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

    # coach_no为01~08，即8编组，还有一种数据为01~16，为16编组，
    # coach_no为车厢。
    # 通过time_value进行分组。
    # 当8编组时，
    # 如果一个时间分组内的3车厢或者6车厢的vcb_status=1，则这一组的vcb_status都填为1，
    # 如果3车厢和6车厢都为2，则这一组的vcb_status都为2.
    # 其他情况就是0，
    # 如果是16编组，
    # 则1~8车厢里面的3车厢或者6车厢的vcb_status=1，则这组的1~8车厢的vcb_status都为1，
    # 如果3车厢和6车厢都为2，则这一组的1~8车厢vcb_status都为2.
    # 其他情况就是0；
    # 9~16车厢，里面的11车厢或者14车厢的vcb_status=1，则这组的9~16车厢的vcb_status都为1，
    # 如果11车厢和14车厢都为2，则这一组的9~16车厢vcb_status都为2.
    # 其他情况就是0

    # 确保 coach_no 是整数类型，以便进行数字比较
    # 如果源数据是 "01", "02" 这种字符串，这步会将其转为 1, 2
    df['coach_no'] = df['coach_no'].astype(int)

    print("开始填充车厢的vcb_status...")

    # df_group_data = df.groupby('time_value')
    # for _, df_group in df_group_data:
    #     # 罗列出分组内的所有车厢编号
    #     vcb_status = df_group['vcb_status'].unique()
    #
    #     one_flag = (1.0 in vcb_status and pd.isna(vcb_status).any())
    #     two_flag = (2.0 in vcb_status and pd.isna(vcb_status).any())
    #
    #     # TODO 待完善区分16编组
    #     if one_flag:
    #         # 分组内所有车厢的vcb_status都为1
    #         df.loc[df_group.index, 'vcb_status'] = 1
    #     elif two_flag and not one_flag:
    #         # 分组内所有车厢的vcb_status都为2
    #         df.loc[df_group.index, 'vcb_status'] = 2
    #     else:
    #         # 分组内所有车厢的vcb_status都为0
    #         df.loc[df_group.index, 'vcb_status'] = 0
 # -------------------------------------------------------------
 #    df_group_data = df.groupby('time_value')

    def optimize_vcb_status(df):
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

        # 3. 更新 vcb_status
        # 将计算出的广播值赋回原列，同时将原本就是 NaN 且没广播到值的（即整组都为空）填为 0
        df['vcb_status'] = filled_vcb.fillna(0)

        # 4. 删除辅助列
        df.drop(columns=['group_id'], inplace=True)

        return df

    # 调用
    df = optimize_vcb_status(df)

    #
    # def process_group(group):
    #     # --- 处理 1-8 车厢段 ---
    #     # 筛选出 1-8 车厢的数据
    #     mask_1_8 = (group['coach_no'] >= 1) & (group['coach_no'] <= 8)
    #
    #     # 如果这个组里有 1-8 车厢的数据
    #     if mask_1_8.any():
    #         subset = group[mask_1_8]
    #         # 获取 3车 和 6车 的状态
    #         # 使用 values[0] 取值，如果车厢不存在则默认为 0
    #         v3_series = subset.loc[subset['coach_no'] == 3, 'vcb_status']
    #         v6_series = subset.loc[subset['coach_no'] == 6, 'vcb_status']
    #
    #         val3 = v3_series.values[0] if not v3_series.empty else 0
    #         val6 = v6_series.values[0] if not v6_series.empty else 0
    #
    #         new_status = 0
    #         # 规则：任一为1则全1
    #         if val3 == 1 or val6 == 1:
    #             new_status = 1
    #         # 规则：都为2则全2
    #         elif val3 == 2 and val6 == 2:
    #             new_status = 2
    #         # 其他情况保持为 0
    #
    #         # 将计算出的新状态赋值给该组内 1-8 车厢的所有行
    #         group.loc[mask_1_8, 'vcb_status'] = new_status
    #
    #     # print("处理完成...")
    #     # --- 处理 9-16 车厢段 ---
    #     # print("开始填充9~16节车厢的vcb_status...")
    #     # 筛选出 9-16 车厢的数据
    #     mask_9_16 = (group['coach_no'] >= 9) & (group['coach_no'] <= 16)
    #
    #     # 如果这个组里有 9-16 车厢的数据
    #     if mask_9_16.any():
    #         subset = group[mask_9_16]
    #         # 获取 11车 和 14车 的状态
    #         v11_series = subset.loc[subset['coach_no'] == 11, 'vcb_status']
    #         v14_series = subset.loc[subset['coach_no'] == 14, 'vcb_status']
    #
    #         val11 = v11_series.values[0] if not v11_series.empty else 0
    #         val14 = v14_series.values[0] if not v14_series.empty else 0
    #
    #         new_status = 0
    #         if val11 == 1 or val14 == 1:
    #             new_status = 1
    #         elif val11 == 2 and val14 == 2:
    #             new_status = 2
    #
    #         group.loc[mask_9_16, 'vcb_status'] = new_status
    #
    #     # print("处理完成...")
    #
    #     return group
    #
    # # 对 time_value 进行分组并应用处理函数
    # # group_keys=False 保持原索引结构
    # # result_df = df.groupby('time_value', group_keys=False).apply(process_group)
    result_df = df.copy()

    print("开始处理温度字段")
    # 清理字段 'charge_battery_group1_tem1', 'charge_battery_group1_tem2', 'charge_battery_group2_tem1', 'charge_battery_group2_tem2'
    # 温度字段
    template_cols = [
        'charge_battery_group1_tem1', 'charge_battery_group1_tem2', 'charge_battery_group2_tem1',
        'charge_battery_group2_tem2'
    ]

    print("删除蓄电池VCB为0的行...")
    result_df = result_df[result_df['vcb_status'] != 0]

    print("删除蓄电池温度都为空的行...")

    # 如果四个列数据都为 Null，则删除该行数据
    result_df = result_df.dropna(subset=template_cols, how='all')

    print("删除温度值为负的行...")
    # 如果四个列数据都以 - 开头，则删除该行数据
    result_df = result_df[~result_df[template_cols].astype(str).apply(lambda x: x.str.startswith('-').all(), axis=1)]

    # 电压字段
    charger_out_cols = ['charger_out_v1', 'charger_out_v2']

    print("删除电压都为空的行...")
    # 如果都为 Null，则删除该行数据
    result_df = result_df.dropna(subset=charger_out_cols, how='all')
    # 如果都是0，则删除该行数据
    print("删除电压都为0的行...")
    result_df = result_df[~result_df[charger_out_cols].astype(str).apply(lambda x: x.eq('0').all(), axis=1)]

    # 只保留 vcb_status = 2 的数据
    # print("只保留 vcb_status = 2 的数据...")
    # result_df = result_df[result_df['vcb_status'] == 2]

    # 将Null置为 NAN
    print("将Null置为 NAN...")
    result_df = result_df.fillna(np.nan)

    # 添加列 charge_battery_group_tem_1 = 蓄电池组1平均温度， charge_battery_group_tem_2 = 蓄电池组2平均温度
    print("添加列 charge_battery_group_tem_1 = 蓄电池组1平均温度， charge_battery_group_tem_2 = 蓄电池组2平均温度...")
    result_df['charge_battery_group_tem_1'] = result_df.apply(
        lambda x: (x['charge_battery_group1_tem1'] + x['charge_battery_group1_tem2']) / 2, axis=1)
    result_df['charge_battery_group_tem_2'] = result_df.apply(
        lambda x: (x['charge_battery_group2_tem1'] + x['charge_battery_group2_tem2']) / 2, axis=1)

    # 先按时间从小到达排序，再按车厢排序，然后将车厢按顺序堆叠
    print("执行排序：时间 -> 车厢...")
    # 将时间转换为字符串或直接利用数值排序，转为Datetime方便后续计算
    # time_value 格式为 YYYYMMDDHHMMSS 的数字或字符串
    result_df = result_df.sort_values(by=['coach_no', 'time_value'], ascending=[True, True])

    # 添加放电时长列，discharge_duration
    # 秒级单位，当time_value跨度超过5分钟就认为是下一组数据，则认为该组数据结束
    # 应为上面已经按照时间从小到大排序，所以每个时间段的第一行数据为0，时间段内数据递增1.
    # 时间段结束后，下一时间段继续从0开始
    # 临时时间对象
    # result_df['temp_datetime'] = pd.to_datetime(result_df['time_value'].astype(str), format='%Y%m%d%H%M%S',
    #                                             errors='coerce')

    # 全局排序：先车厢，再时间
    result_df = result_df.sort_values(by=['coach_no', 'time_value'], ascending=[True, True])

    # def mark_discharge_groups(group):
    #     """
    #     为每个车厢生成“事件ID” (segment_id)
    #     分段条件：
    #     1. VCB状态发生变化 (例如 2->0 或 0->2)
    #     2. 时间跨度超过 5 分钟 (断数)
    #     """
    #     # 计算时间差
    #     time_diff = group['temp_datetime'].diff()
    #
    #     # 条件1：时间断层 > 5分钟
    #     is_time_gap = time_diff > pd.Timedelta(minutes=5)
    #
    #     # 条件2：VCB状态变化 (当前状态 != 上一行的状态)
    #     is_status_change = group['vcb_status'] != group['vcb_status'].shift()
    #
    #     # 只要满足任一条件，就视为新的一段
    #     new_segment = is_time_gap | is_status_change
    #
    #     # 生成段ID
    #     group['segment_id'] = new_segment.cumsum()
    #
    #     return group

    # 按车厢分组生成标记
    # result_df = result_df.groupby('coach_no', group_keys=False).apply(mark_discharge_groups)

    # =========================================================
    # 数据清洗 (现在可以安全地删除行了，因为 segment_id 已经生成)
    # =========================================================
    print("正在清洗温度和电压数据...")

    # 清洗温度
    temp_cols = ['charge_battery_group1_tem1', 'charge_battery_group1_tem2', 'charge_battery_group2_tem1',
                 'charge_battery_group2_tem2']
    result_df = result_df.dropna(subset=temp_cols, how='all')
    result_df = result_df[~result_df[temp_cols].astype(str).apply(lambda x: x.str.startswith('-').all(), axis=1)]

    # 清洗电压
    volt_cols = ['charger_out_v1', 'charger_out_v2']
    result_df = result_df.dropna(subset=volt_cols, how='all')
    result_df = result_df[~result_df[volt_cols].astype(str).apply(lambda x: x.eq('0').all(), axis=1)]

    # 过滤：只保留 VCB=2 (放电) 的数据
    print("过滤非放电数据...")
    # result_df = result_df[result_df['vcb_status'] == 2]
    # result_df = result_df[result_df['vcb_status'] in [1, 2]]

    # 填充空值
    result_df = result_df.fillna(np.nan)

    # 计算平均温度
    result_df['charge_battery_group_tem_1'] = result_df[
        ['charge_battery_group1_tem1', 'charge_battery_group1_tem2']].mean(axis=1)
    result_df['charge_battery_group_tem_2'] = result_df[
        ['charge_battery_group2_tem1', 'charge_battery_group2_tem2']].mean(axis=1)

    # def calculate_duration_by_segment(group):
    #     # 这个 group 是同一个车厢、同一个连续放电段的数据
    #     # 所以起始时间就是这个组的最小时间
    #     start_time = group['temp_datetime'].min()
    #     group['discharge_duration'] = (group['temp_datetime'] - start_time).dt.total_seconds()
    #     return group

    # # 这里需要按 ['coach_no', 'segment_id'] 分组，确保每段独立计算
    # if not result_df.empty:
    #     result_df = result_df.groupby(['coach_no', 'segment_id'], group_keys=False).apply(calculate_duration_by_segment)
    # else:
    #     result_df['discharge_duration'] = 0

    # 最终排序：车厢 -> 时间
    print("执行最终排序...")
    result_df = result_df.sort_values(by=['coach_no', 'time_value'], ascending=[True, True])

    # 清理临时列
    cols_to_drop = ['temp_datetime', 'segment_id']
    result_df = result_df.drop(columns=[c for c in cols_to_drop if c in result_df.columns])

    # =========================================================
    # 计算放电时长 (基于之前生成的 segment_id)
    # =========================================================
    print("计算放电时长...")
    # 添加字段discharge_duration

    # 从第一行开始，铆钉字段time_value，第一行为0，接着往下+1，当time_value不连续时，则认为该段结束，不连续的下一行为下一段数据，从0开始
    # 初始化 discharge_duration
    discharge_duration = [0]

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

    # 检查并保留字段
    # final_cols = [c for c in keep_columns if c in result_df.columns]

    # return result_df[final_cols]


    # 电池组1、2
    final_cols_1 = [c for c in charge_group_1_columns if c in result_df.columns]
    final_cols_2 = [c for c in charge_group_2_columns if c in result_df.columns]

    result_df_1 = result_df[final_cols_1]
    result_df_2 = result_df[final_cols_2]

    # 对两组数据根据time_value进行去重
    # coach_no = 1的数据
    result_df_1

    # return result_df[final_cols_1], result_df[final_cols_2]
    return result_df_1, result_df_2


def read_and_save_as_csv(file_path, save_path_1, save_path_2):
    try:
        df_1, df_2 = read_csv(file_path)
        # df.to_csv(save_path, index=False)
        df_1.to_csv(save_path_1, index=False)
        df_2.to_csv(save_path_2, index=False)

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

    dir_path = r"D:\Work\LLM\GitHub\Large-Time-Series-Model\csv_data\original_data"
    clean_dir_path = r"D:\Work\LLM\GitHub\Large-Time-Series-Model\csv_data\clean_data"
    clean_txt_dir_path = r"D:\Work\LLM\GitHub\Large-Time-Series-Model\dataset\xudianchi\train\normal"

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

    # 将 csv 文件转换为 txt 文件
    for file in tqdm(os.listdir(clean_dir_path)):
        file_path = os.path.join(clean_dir_path, file)
        csv_to_txt(file_path, clean_txt_dir_path)

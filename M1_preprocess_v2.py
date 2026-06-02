"""
M1_preprocess_v2: Upgrade to preserve 3D spatial structure
Output format: (T_steps, S_stations=5, F_features=8) as .npy files
Prevents "space-aware" encoding from collapsing into single modality
"""
import pandas as pd
import numpy as np
import os
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置参数（复用原有）
# ============================================================
RAW_DATA_PATH = r"E:\Desktop\zohu\data"
OUTPUT_PATH = r"E:\Desktop\zohu\data\processed"

STATIONS = {
    '02': (21.517167, 108.213667),
    '14': (21.547500, 108.920167),
    '15': (21.510833, 109.043017),
    '17': (21.411019, 109.244914),
    '19': (21.458833, 109.540833),
}

CORE_FEATURES = [
    'Temp_C', 'Sal', 'DO_ppm', 'DO_percent', 'pH', 'Turb_NTU', 'Chl_ppb', 'PE_uL',
]

PHYSICAL_BOUNDS = {
    'Temp_C': (-2, 35),
    'Sal': (10, 45),
    'DO_percent': (30, 300),
    'DO_ppm': (3, 20),
    'pH': (6.8, 8.8),
    'Turb_NTU': (0, 1000),
    'Chl_ppb': (0, 400),
    'PE_uL': (0, 200000),
}

FILE_NAMES = {
    '02': 'GX-02_2019.csv',
    '14': 'GX-14_2019.csv',
    '15': 'GX-15_2019.csv',
    '17': 'GX-17_2019.csv',
    '19': 'GX-19_2019.csv',
}

KNOWN_DISTANCES_MILES = {
    ('02', '14'): 45.96, ('02', '15'): 45.34, ('02', '17'): 66.99, ('02', '19'): 86.11,
    ('14', '15'): 8.35,  ('14', '17'): 22.92, ('14', '19'): 40.37,
    ('15', '17'): 14.70, ('15', '19'): 32.28,
    ('17', '19'): 19.64,
}

TIME_RESOLUTION = '30min'


# ============================================================
# 核心函数（复用原有逻辑）
# ============================================================

def detect_and_classify_anomalies_improved(df, station_id):
    """异常分类逻辑（复用）"""
    df_analysis = df.copy()
    df_analysis['is_fault'] = False
    df_analysis['anomaly_type'] = 'normal'
    df_analysis['fault_reason'] = ''

    CRITICAL_FEATURES = ['Temp_C', 'Sal', 'pH']
    for idx in df_analysis.index:
        is_critical_fault = False
        fault_features = []
        for col in CRITICAL_FEATURES:
            if col in df_analysis.columns:
                val = df_analysis.loc[idx, col]
                low, high = PHYSICAL_BOUNDS.get(col, (0, 1000))
                if pd.notna(val) and (val < low or val > high):
                    is_critical_fault = True
                    fault_features.append(f'{col}={val:.1f}')
        if is_critical_fault:
            df_analysis.loc[idx, 'is_fault'] = True
            df_analysis.loc[idx, 'anomaly_type'] = 'sensor_fault'
            df_analysis.loc[idx, 'fault_reason'] = f'关键参数异常: {fault_features}'

    df_analysis['Chl_prev'] = df_analysis['Chl_ppb'].shift(1)
    df_analysis['DO_prev'] = df_analysis['DO_ppm'].shift(1)
    for idx in df_analysis.index:
        if df_analysis.loc[idx, 'anomaly_type'] == 'sensor_fault':
            continue
        temp = df_analysis.loc[idx, 'Temp_C']
        sal = df_analysis.loc[idx, 'Sal']
        ph = df_analysis.loc[idx, 'pH']
        if pd.isna(temp) or pd.isna(sal) or pd.isna(ph):
            continue
        temp_ok = (temp >= 0 and temp <= 35)
        sal_ok = (sal >= 20 and sal <= 45)
        ph_ok = (ph >= 7 and ph <= 9)
        if not (temp_ok and sal_ok and ph_ok):
            continue
        chl = df_analysis.loc[idx, 'Chl_ppb']
        chl_prev = df_analysis.loc[idx, 'Chl_prev']
        do = df_analysis.loc[idx, 'DO_ppm']
        do_prev = df_analysis.loc[idx, 'DO_prev']
        if pd.isna(chl) or pd.isna(chl_prev):
            continue
        if chl_prev > 0:
            chl_condition = (chl / chl_prev) >= 2.5
        else:
            chl_condition = chl >= 5.0
        do_condition = True
        if pd.notna(do) and pd.notna(do_prev) and do_prev > 0:
            do_condition = do / do_prev >= 1.1
        if chl_condition and do_condition:
            df_analysis.loc[idx, 'anomaly_type'] = 'bloom_event'
            df_analysis.loc[idx, 'is_fault'] = False

    df_analysis = df_analysis.drop(columns=['Chl_prev', 'DO_prev', 'pH_prev'], errors='ignore')
    return df_analysis


def handle_faults_by_type(df, station_id, bounds):
    """故障处理（复用）"""
    print(f"\n  站点 {station_id} 异常检测与分类:")
    df_processed = detect_and_classify_anomalies_improved(df, station_id)
    fault_mask = df_processed['is_fault']
    n_fault = fault_mask.sum()
    if n_fault > 0:
        for col in CORE_FEATURES:
            if col in df_processed.columns:
                df_processed.loc[fault_mask, col] = np.nan
        print(f"    处理方式: {n_fault} 个故障点已被设为NaN")
    return df_processed


def load_and_align_station(filepath, station_id):
    """加载、清洗、对齐单个站点"""
    df = pd.read_csv(filepath)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    df = df.sort_values('TIMESTAMP').reset_index(drop=True)
    print(f"  加载站点 {station_id}: {df.shape[0]} 行，时间范围 {df['TIMESTAMP'].min()} ~ {df['TIMESTAMP'].max()}")

    # 故障检测
    df = handle_faults_by_type(df, station_id, PHYSICAL_BOUNDS)

    # 时间对齐
    df['TIMESTAMP_ROUNDED'] = df['TIMESTAMP'].dt.round(TIME_RESOLUTION)
    agg_dict = {col: 'mean' for col in CORE_FEATURES if col in df.columns}
    df = df.groupby('TIMESTAMP_ROUNDED').agg(agg_dict).reset_index()
    df = df.rename(columns={'TIMESTAMP_ROUNDED': 'TIMESTAMP'})

    start_time = df['TIMESTAMP'].min()
    end_time = df['TIMESTAMP'].max()
    full_time_grid = pd.date_range(start=start_time, end=end_time, freq=TIME_RESOLUTION)
    df = df.set_index('TIMESTAMP').reindex(full_time_grid).reset_index()
    df = df.rename(columns={'index': 'TIMESTAMP'})
    df['Station'] = station_id

    print(f"    对齐后: {df.shape[0]} 行")
    return df


def main():
    print("=" * 70)
    print("M1_preprocess_v2: Spatial 3D Structure Output")
    print("=" * 70)

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # ===== 第一步：加载、清洗、对齐所有站点 =====
    print("\n[1/3] 加载与对齐所有站点...")
    station_dfs = {}
    for station_id, filename in FILE_NAMES.items():
        filepath = os.path.join(RAW_DATA_PATH, filename)
        if not os.path.exists(filepath):
            print(f"  ⚠ 站点 {station_id} 数据文件不存在")
            continue
        try:
            df = load_and_align_station(filepath, station_id)
            station_dfs[station_id] = df
        except Exception as e:
            print(f"  ⚠ 站点 {station_id} 加载失败: {e}")

    if len(station_dfs) == 0:
        print("错误: 没有成功加载任何站点")
        return

    print(f"\n✓ 成功加载 {len(station_dfs)} 个站点")

    # ===== 第二步：时间序列对齐 =====
    print("\n[2/3] 统一时间对齐...")
    all_timestamps = set()
    for df in station_dfs.values():
        all_timestamps.update(df['TIMESTAMP'])
    all_timestamps = sorted(all_timestamps)
    print(f"  统一时间轴: {len(all_timestamps)} 个时间点")

    # 重新���引到统一时间轴
    for station_id in station_dfs:
        df = station_dfs[station_id]
        df = df.set_index('TIMESTAMP').reindex(all_timestamps).reset_index()
        df = df.rename(columns={'index': 'TIMESTAMP'})
        station_dfs[station_id] = df

    # ===== 第三步：构建 3D 张量 (T, S, F) =====
    print("\n[3/3] 构建 3D 空间张量...")
    station_order = sorted(station_dfs.keys())
    T = len(all_timestamps)
    S = len(station_order)
    F = len(CORE_FEATURES)

    # 初始化 3D 数组
    data_3d = np.full((T, S, F), np.nan, dtype=np.float32)
    valid_mask_3d = np.zeros((T, S, F), dtype=bool)

    # 填充数据
    for s, station_id in enumerate(station_order):
        df = station_dfs[station_id]
        for f, feature in enumerate(CORE_FEATURES):
            if feature in df.columns:
                col_data = df[feature].values
                data_3d[:, s, f] = col_data
                valid_mask_3d[:, s, f] = ~np.isnan(col_data)

    print(f"  3D张量形状: {data_3d.shape}")
    print(f"  总缺失率: {(~valid_mask_3d).sum() / valid_mask_3d.size * 100:.2f}%")
    print(f"  按站点缺失率:")
    for s, station_id in enumerate(station_order):
        missing_rate = (~valid_mask_3d[:, s, :]).sum() / (T * F) * 100
        print(f"    站点 {station_id}: {missing_rate:.2f}%")

    # ===== 第四步：保存输出 =====
    print("\n  保存 3D 数据结构...")

    # 保存 3D 数据和掩码
    np.save(os.path.join(OUTPUT_PATH, "M1_data_3d.npy"), data_3d)
    np.save(os.path.join(OUTPUT_PATH, "M1_valid_mask_3d.npy"), valid_mask_3d)

    # 保存元数据
    metadata = {
        'stations': station_order,
        'features': CORE_FEATURES,
        'timestamps': [str(t) for t in all_timestamps],
        'shape': (T, S, F),
    }
    import json
    with open(os.path.join(OUTPUT_PATH, "M1_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    # 保存距离矩阵
    dist_matrix = np.zeros((S, S))
    for i, s1 in enumerate(station_order):
        for j, s2 in enumerate(station_order):
            if i == j:
                continue
            key = (min(s1, s2), max(s1, s2))
            miles = KNOWN_DISTANCES_MILES.get(key, 0.0)
            dist_matrix[i, j] = miles * 1.60934
    np.save(os.path.join(OUTPUT_PATH, "M1_distance_matrix.npy"), dist_matrix)

    # 保存兼容的 2D 宽表（用于 PCMCI+）
    df_wide = []
    for s, station_id in enumerate(station_order):
        df_station = station_dfs[station_id][['TIMESTAMP'] + CORE_FEATURES].copy()
        df_station.columns = ['TIMESTAMP'] + [f"{f}_{station_id}" for f in CORE_FEATURES]
        if len(df_wide) == 0:
            df_wide = df_station
        else:
            df_wide = pd.merge(df_wide, df_station, on='TIMESTAMP', how='outer')

    df_wide.to_csv(os.path.join(OUTPUT_PATH, "M1_wide_table.csv"), index=False)

    print("\n✓ 所有文件已保存:")
    print(f"  • M1_data_3d.npy: 3D 空间张量 {data_3d.shape}")
    print(f"  • M1_valid_mask_3d.npy: 有效性掩码")
    print(f"  • M1_metadata.json: 元数据")
    print(f"  • M1_distance_matrix.npy: 站点距离矩阵")
    print(f"  • M1_wide_table.csv: PCMCI+ 兼容宽表")

    print("\n" + "=" * 70)
    print("M1 升级完成！下一步：M2 PCMCI+ 因果发现")
    print("=" * 70)


if __name__ == "__main__":
    main()

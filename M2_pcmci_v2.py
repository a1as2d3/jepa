"""
M2_pcmci+_v2: Causal Discovery with Spatial Aggregation
Outputs both fine-grained (40x40) and spatial (5x5) causal masks
"""
import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tqdm import tqdm

# ============================================================
# 配置参数
# ============================================================
PROCESSED_DATA_PATH = r"E:\Desktop\zohu\data\processed"
INPUT_FILE = os.path.join(PROCESSED_DATA_PATH, "M1_wide_table.csv")

TAU_MIN = 0
TAU_MAX = 6
PC_ALPHA = 0.001

# ============================================================
# 数据准备（无泄漏）
# ============================================================
def prepare_data_for_pcmci(df):
    """绝对无泄漏数据准备"""
    original_missing_rate = df.isnull().sum() / len(df)
    excluded_cols = original_missing_rate[original_missing_rate > 0.20].index.tolist()

    df_forward = df.ffill()
    df_stat = df_forward.diff().dropna()

    if excluded_cols:
        print(f"  排除高缺失率变量: {excluded_cols}")
        df_stat = df_stat.drop(columns=excluded_cols, errors='ignore')

    return df_stat, excluded_cols


def run_pcmciplus_on_segment(data_array, var_names, verbosity=1):
    """运行 PCMCI+"""
    T, N = data_array.shape
    dataframe = pp.DataFrame(data_array, datatime=np.arange(T), var_names=var_names)
    parcorr = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=verbosity)
    results = pcmci.run_pcmciplus(tau_min=TAU_MIN, tau_max=TAU_MAX, pc_alpha=PC_ALPHA)
    return results


# ============================================================
# 因果先验计算（带矩阵转置对齐）
# ============================================================
def compute_causal_prior(p_matrix, val_matrix, pc_alpha, var_names=None):
    """
    Tigramite 约定: matrix[i, j] 代表 i -> j
    PyTorch Attention 约定: attn_mask[target, source] 代表 source -> target
    因此进行 .T 转置对齐
    """
    N = p_matrix.shape[0]

    # 提取最强效应
    min_p_raw = np.min(p_matrix, axis=2)
    best_tau_raw = np.argmax(np.abs(val_matrix), axis=2)
    binary_mask_raw = (min_p_raw < pc_alpha).astype(int)

    # 转置：[Source, Target] -> [Target, Source]
    binary_mask = binary_mask_raw.T
    min_p = min_p_raw.T
    best_tau = best_tau_raw.T

    # 填对角线
    np.fill_diagonal(binary_mask, 1)

    # 软先验
    with np.errstate(divide='ignore'):
        log_p = -np.log(np.clip(min_p, 1e-10, 1.0))
    soft_prior = np.where(binary_mask == 1, log_p, 0.0)
    diag_val = soft_prior.max() if soft_prior.max() > 0 else 1.0
    np.fill_diagonal(soft_prior, diag_val)
    soft_prior = soft_prior / (soft_prior.max() + 1e-8)

    # PyTorch 注意力掩码
    attn_mask = (binary_mask == 0)
    np.fill_diagonal(attn_mask, False)

    return binary_mask, soft_prior, best_tau, attn_mask


# ============================================================
# 核心创新：空间因果聚合
# ============================================================
def aggregate_to_spatial_causal_mask(binary_mask_40x40, var_names, stations, features, aggregation='any'):
    """
    将 40x40 细粒度因果图聚合到 5x5 空间因果图
    
    参数：
    - binary_mask_40x40: (40, 40) 变量因果矩阵，binary_mask[j, i]=1 表示 i->j
    - var_names: 40 个变量名列表
    - stations: 5 个站点ID列表
    - features: 8 个特征名列表
    - aggregation: 'any'(任意边存在) | 'majority'(大多数边) | 'strength'(总强度)
    
    返回：
    - spatial_mask_5x5: (5, 5) 站点因果矩阵
    - edge_counts: (5, 5) 各区块边数统计
    """
    S = len(stations)
    F = len(features)
    N_vars = S * F

    if binary_mask_40x40.shape != (N_vars, N_vars):
        raise ValueError(f"Expected shape ({N_vars}, {N_vars}), got {binary_mask_40x40.shape}")

    spatial_mask = np.zeros((S, S), dtype=int)
    edge_counts = np.zeros((S, S), dtype=int)

    for s_tgt in range(S):
        for s_src in range(S):
            if s_tgt == s_src:
                # 自环：只要任何特征对存在边，就认为存在自环
                edge_count = 0
                for f_tgt in range(F):
                    for f_src in range(F):
                        idx_tgt = s_tgt * F + f_tgt
                        idx_src = s_src * F + f_src
                        if binary_mask_40x40[idx_tgt, idx_src] == 1:
                            edge_count += 1
                if edge_count > 0:
                    spatial_mask[s_tgt, s_src] = 1
                edge_counts[s_tgt, s_src] = edge_count
            else:
                # 跨站边：从 s_src 的所有特征指向 s_tgt 的所有特征
                edge_count = 0
                for f_tgt in range(F):
                    for f_src in range(F):
                        idx_tgt = s_tgt * F + f_tgt
                        idx_src = s_src * F + f_src
                        if binary_mask_40x40[idx_tgt, idx_src] == 1:
                            edge_count += 1

                if aggregation == 'any':
                    spatial_mask[s_tgt, s_src] = 1 if edge_count > 0 else 0
                elif aggregation == 'majority':
                    spatial_mask[s_tgt, s_src] = 1 if edge_count >= (F // 2) else 0
                elif aggregation == 'strength':
                    spatial_mask[s_tgt, s_src] = 1 if edge_count >= 3 else 0

                edge_counts[s_tgt, s_src] = edge_count

    return spatial_mask, edge_counts


# ============================================================
# 主程序
# ============================================================
def run_causal_discovery_v2():
    print("=" * 70)
    print("M2_pcmci+_v2: Causal Discovery with Spatial Aggregation")
    print("=" * 70)

    # 加载元数据
    with open(os.path.join(PROCESSED_DATA_PATH, "M1_metadata.json"), 'r') as f:
        metadata = json.load(f)
    stations = metadata['stations']
    features = metadata['features']
    print(f"\n站点: {stations}")
    print(f"特征: {features}")

    # 加载宽表数据
    df = pd.read_csv(INPUT_FILE)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    df = df.set_index('TIMESTAMP')

    # 数据准备
    df_stat, excluded_cols = prepare_data_for_pcmci(df)
    data_array = df_stat.values
    T, N = data_array.shape
    var_names = df_stat.columns.tolist()

    print(f"\n数据规模: {T} 时间点 × {N} 变量")

    # ===== 1. 全局静态因果图 =====
    print("\n[1/2] 运行全局 PCMCI+...")
    results_global = run_pcmciplus_on_segment(data_array, var_names, verbosity=1)
    binary_mask, soft_prior, best_tau, attn_mask = compute_causal_prior(
        results_global['p_matrix'], results_global['val_matrix'], PC_ALPHA, var_names
    )

    n_edges = binary_mask.sum() - N
    print(f"  发现因果边数: {n_edges} 条")

    # 保存细粒度因果图
    np.save(os.path.join(PROCESSED_DATA_PATH, "M2_causal_mask.npy"), binary_mask)
    np.save(os.path.join(PROCESSED_DATA_PATH, "M2_soft_prior.npy"), soft_prior)
    np.save(os.path.join(PROCESSED_DATA_PATH, "M2_best_tau.npy"), best_tau)
    np.save(os.path.join(PROCESSED_DATA_PATH, "M2_attn_mask.npy"), attn_mask)

    # ===== 2. 空间聚合：从 40x40 -> 5x5 =====
    print("\n[2/2] 空间因果聚合 (40x40 -> 5x5)...")
    spatial_mask, edge_counts = aggregate_to_spatial_causal_mask(
        binary_mask, var_names, stations, features, aggregation='any'
    )

    print(f"\n  空间因果图 (5x5):")
    print(f"  行=目标站点, 列=源站点")
    print(f"  {spatial_mask}")

    print(f"\n  各区块边数统计:")
    for s_tgt in range(len(stations)):
        for s_src in range(len(stations)):
            count = edge_counts[s_tgt, s_src]
            if count > 0:
                print(f"    {stations[s_src]:2s} → {stations[s_tgt]:2s}: {count:2d} 条边")

    # 构建空间注意力掩码（与 PyTorch Attention 兼容）
    spatial_attn_mask = (spatial_mask == 0).astype(bool)
    np.fill_diagonal(spatial_attn_mask, False)

    # 保存空间因果图
    np.save(os.path.join(PROCESSED_DATA_PATH, "M2_spatial_causal_mask.npy"), spatial_mask)
    np.save(os.path.join(PROCESSED_DATA_PATH, "M2_spatial_attn_mask.npy"), spatial_attn_mask)

    # 生成可视化报告
    report_path = os.path.join(PROCESSED_DATA_PATH, "M2_causal_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("M2 PCMCI+ 因果发现报告\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"全局因果图统计:\n")
        f.write(f"  变量数: {N}\n")
        f.write(f"  因果边数: {n_edges}\n")
        f.write(f"  密度: {n_edges / (N * (N - 1)):.4f}\n\n")
        
        f.write(f"空间因果图 (5x5):\n")
        f.write(f"  站点: {stations}\n")
        f.write(f"  邻接矩阵 (1=有因果):\n")
        for row in spatial_mask:
            f.write("  " + " ".join(str(int(x)) for x in row) + "\n")
        
        f.write(f"\n各区块边数:\n")
        for s_tgt in range(len(stations)):
            for s_src in range(len(stations)):
                count = edge_counts[s_tgt, s_src]
                if count > 0:
                    f.write(f"  {stations[s_src]:2s} → {stations[s_tgt]:2s}: {count:2d}\n")

    print(f"\n✓ 因果报告保存至: {report_path}")

    print("\n" + "=" * 70)
    print("M2 升级完成！")
    print("=" * 70)


if __name__ == "__main__":
    run_causal_discovery_v2()

"""
Ocean Water Quality Data Repair using C-JEPA Architecture
===========================================================

数据: 海洋水质传感器数据
形状: (17520, 40) - 12个监测点 × 8个参数 + 时间索引
缺失率: 5.93%
问题: 传感器故障导致的数据缺失和异常

改造: 用C-JEPA的对象级掩码预测 → 修复时空相关的水质参数

核心思想:
├─ 原始C-JEPA: 掩码对象轨迹 → 从邻域对象推断
├─ 本框架: 掩码水质参数 → 从邻域监测点和时间序列推断
└─ 关键差异: 从视频帧的空间相邻性 转变为 传感器网络的拓扑相邻性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math
import random
from typing import Tuple, Dict, List

# ============================================================================
# 配置参数 (类似 C-JEPA 的常数)
# ============================================================================

SENSOR_LOCATIONS = {
    "02": (0, 2), "14": (1, 4), "15": (1, 5), "17": (1, 7), "19": (1, 9),  # 5个内湾传感器
}  # 后续可扩展为真实地理坐标

PARAMS = [
    "Temp_C",      # 温度 (℃)
    "Sal",         # 盐度 (PSU)
    "DO_ppm",      # 溶解氧 (ppm)
    "DO_percent",  # 溶解氧饱和度 (%)
    "pH",          # pH值
    "Turb_NTU",    # 浊度 (NTU)
    "Chl_ppb",     # 叶绿素 (ppb)
    "PE_uL",       # 叶绿素荧光 (uL)
]

N_SENSORS = len(SENSOR_LOCATIONS)      # 5
N_PARAMS = len(PARAMS)                 # 8
N_CHANNELS = N_SENSORS * N_PARAMS      # 40
SENSOR_DIM = 64                        # 嵌入维度 (C-JEPA用128, 这里用64)

# 时间分割 (类似 C-JEPA 的 T_HIST, T_PRED)
WINDOW_LEN = 48                        # 24小时 (30分钟间隔 = 48步)
HISTORY_LEN = 36                       # 18小时 (历史上下文)
FUTURE_LEN = 12                        # 6小时  (预测未来)


# ============================================================================
# 第1部分: 数据加载与预处理
# ============================================================================

class OceanDataRepairDataset(Dataset):
    """
    海洋水质数据修复数据集
    
    改造自 C-JEPA 的 BouncingTriple:
    ├─ C-JEPA: 合成视频 + 已知的对象位置
    └─ 本框架: 真实传感器数据 + 模拟的缺失区域
    
    关键改动:
    - 输入: (完整时间序列, 损坏版本, 缺失掩码)
    - 缺失模式: 传感器故障、网络中断、异常值
    """
    
    def __init__(self, 
                 data_path: str = None,
                 window_len: int = WINDOW_LEN,
                 history_len: int = HISTORY_LEN,
                 future_len: int = FUTURE_LEN,
                 corruption_type: str = "sensor_failure",
                 corruption_ratio: float = 0.1,
                 n_samples: int = 1000,
                 train: bool = True,
                 seed: int = 42):
        """
        corruption_type:
        ├─ "sensor_failure"      : 整个传感器故障 (某传感器某段时间全缺)
        ├─ "param_missing"       : 单个参数缺失 (温度传感器故障, 盐度正常)
        ├─ "temporal_gap"        : 时间中断 (网络故障导致某时刻无数据)
        └─ "mixed"               : 混合缺陷
        """
        self.window_len = window_len
        self.history_len = history_len
        self.future_len = future_len
        self.corruption_type = corruption_type
        self.corruption_ratio = corruption_ratio
        self.n_samples = n_samples
        
        # 生成或加载数据
        if data_path is None:
            self.data = self._generate_synthetic_data(n_samples, seed)
        else:
            self.data = self._load_real_data(data_path, train, seed)
        
        # 数据标准化
        self.data_normalized = self._normalize(self.data)
        
        # 生成缺失掩码
        torch.manual_seed(seed)
        self.missing_masks = self._generate_missing_masks(seed)
    
    def _generate_synthetic_data(self, n_samples: int, seed: int) -> np.ndarray:
        """
        生成合成数据 (类似 C-JEPA 的 BouncingTriple)
        
        特点:
        ├─ 多个传感器间的时间相关性 (温度缓慢变化)
        ├─ 空间相关性 (相邻传感器值接近)
        └─ 周期性 (日循环, 潮汐)
        """
        rng = np.random.RandomState(seed)
        
        # 生成时间序列: (n_samples, window_len, n_channels)
        data = np.zeros((n_samples, self.window_len, N_CHANNELS))
        
        for sample_idx in range(n_samples):
            # 为每个传感器生成基础值
            sensor_bases = {}
            for sensor_id, (lat, lon) in SENSOR_LOCATIONS.items():
                # 基础值: 温度20±5℃, 盐度30±5 PSU等
                base_values = np.array([
                    20 + rng.randn() * 5,      # Temp
                    30 + rng.randn() * 5,      # Sal
                    8 + rng.randn() * 2,       # DO_ppm
                    90 + rng.randn() * 10,     # DO_percent
                    8 + rng.randn() * 0.5,     # pH
                    1 + abs(rng.randn()),      # Turb_NTU
                    5 + abs(rng.randn()),      # Chl_ppb
                    100 + abs(rng.randn()),    # PE_uL
                ])
                sensor_bases[sensor_id] = base_values
            
            # 时间轨迹: 24小时循环
            t_indices = np.arange(self.window_len)
            
            for sensor_idx, sensor_id in enumerate(sorted(SENSOR_LOCATIONS.keys())):
                base = sensor_bases[sensor_id]
                
                # 日周期变化 (温度早晨冷, 下午热)
                daily_cycle = 2 * np.sin(2 * np.pi * t_indices / 48)
                
                # 随机游走 (缓慢变化)
                random_walk = np.cumsum(rng.randn(self.window_len) * 0.1)
                
                for param_idx in range(N_PARAMS):
                    channel_idx = sensor_idx * N_PARAMS + param_idx
                    
                    if param_idx == 0:  # 温度最敏感
                        data[sample_idx, :, channel_idx] = (
                            base[param_idx] + daily_cycle + random_walk
                        )
                    else:  # 其他参数变化较缓
                        data[sample_idx, :, channel_idx] = (
                            base[param_idx] + 0.1 * random_walk
                        )
        
        return data
    
    def _load_real_data(self, data_path: str, train: bool, seed: int) -> np.ndarray:
        """加载真实 CSV 数据"""
        df = pd.read_csv(data_path)
        
        # 提取数值列 (排除时间戳)
        numeric_cols = [col for col in df.columns if col != 'timestamp']
        data = df[numeric_cols].values  # (17520, 40)
        
        # 按 80/20 分割
        rng = np.random.RandomState(seed)
        n_total = data.shape[0]
        indices = rng.permutation(n_total)
        split_idx = int(0.8 * n_total)
        
        if train:
            selected_idx = indices[:split_idx]
        else:
            selected_idx = indices[split_idx:]
        
        # 滑动窗口生成样本
        samples = []
        for i in range(0, len(selected_idx) - self.window_len, self.window_len // 2):
            window_data = data[selected_idx[i:i+self.window_len]]
            samples.append(window_data)
        
        return np.array(samples)
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """标准化到 [-1, 1] (类似 C-JEPA 的 canvas - 0.5)"""
        self.data_mean = data.mean(axis=(0, 1), keepdims=True)
        self.data_std = data.std(axis=(0, 1), keepdims=True) + 1e-6
        
        return (data - self.data_mean) / self.data_std
    
    def _generate_missing_masks(self, seed: int) -> torch.Tensor:
        """
        生成缺失掩码
        
        改造自 C-JEPA 的掩码策略:
        ├─ C-JEPA: is_q[1:T_HIST, mask_indices] (时间+对象维度)
        └─ 本框架: is_missing[某时段, 某传感器或参数] (时间+空间维度)
        """
        n_samples = self.data_normalized.shape[0]
        masks = []
        
        for _ in range(n_samples):
            mask = torch.zeros(self.window_len, N_CHANNELS)
            
            if self.corruption_type == "sensor_failure":
                # 某个传感器在某个时间段故障
                sensor_idx = np.random.randint(0, N_SENSORS)
                start_t = np.random.randint(self.history_len, 
                                           self.window_len - self.future_len)
                duration = np.random.randint(4, 12)  # 2-6小时
                
                param_start = sensor_idx * N_PARAMS
                param_end = param_start + N_PARAMS
                mask[start_t:min(start_t+duration, self.window_len), 
                     param_start:param_end] = 1
            
            elif self.corruption_type == "param_missing":
                # 某个参数在某时间缺失 (e.g. 温度传感器故障)
                param_idx = np.random.randint(0, N_PARAMS)
                sensor_idx = np.random.randint(0, N_SENSORS)
                start_t = np.random.randint(self.history_len, 
                                           self.window_len - self.future_len)
                duration = np.random.randint(6, 18)
                
                channel_idx = sensor_idx * N_PARAMS + param_idx
                mask[start_t:min(start_t+duration, self.window_len), 
                     channel_idx] = 1
            
            elif self.corruption_type == "temporal_gap":
                # 整个监测网络在某时刻中断
                start_t = np.random.randint(self.history_len, 
                                           self.window_len - self.future_len)
                duration = np.random.randint(1, 4)  # 30分钟-2小时
                mask[start_t:min(start_t+duration, self.window_len), :] = 1
            
            elif self.corruption_type == "mixed":
                # 混合: 先故障某传感器, 再中断网络
                n_defects = np.random.randint(1, 3)
                for _ in range(n_defects):
                    if np.random.rand() < 0.5:
                        # 传感器故障
                        sensor_idx = np.random.randint(0, N_SENSORS)
                        start_t = np.random.randint(self.history_len, 
                                                   self.window_len - self.future_len)
                        duration = np.random.randint(3, 10)
                        param_start = sensor_idx * N_PARAMS
                        param_end = param_start + N_PARAMS
                        mask[start_t:min(start_t+duration, self.window_len), 
                             param_start:param_end] = 1
                    else:
                        # 时间中断
                        start_t = np.random.randint(self.history_len, 
                                                   self.window_len - self.future_len)
                        duration = np.random.randint(1, 3)
                        mask[start_t:min(start_t+duration, self.window_len), :] = 1
            
            masks.append(mask)
        
        return torch.stack(masks)  # (n_samples, window_len, n_channels)
    
    def __len__(self) -> int:
        return self.data_normalized.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回 (完整数据, 损坏数据, 缺失掩码)
        
        对标 C-JEPA:
        ├─ C-JEPA: (video, slot_idx, labels)
        └─ 本框架: (clean, corrupted, missing_mask)
        """
        clean = torch.from_numpy(self.data_normalized[idx]).float()
        
        # 创建损坏版本
        missing_mask = self.missing_masks[idx]
        corrupted = clean.clone()
        corrupted[missing_mask > 0.5] = 0  # 缺失处置为0
        
        # 可选: 添加噪声
        noise = torch.randn_like(corrupted) * 0.05
        corrupted = corrupted + noise * missing_mask
        
        return clean, corrupted, missing_mask


# ============================================================================
# 第2部分: 模型 (改造自 C-JEPA)
# ============================================================================

def sincos_1d(n: int, dim: int) -> torch.Tensor:
    """时间位置编码 (从C-JEPA复用)"""
    pos = torch.arange(n).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim, 2).float() * 
                    (-math.log(10000.) / dim))
    pe = torch.zeros(n, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class Block(nn.Module):
    """标准 Transformer 块 (从C-JEPA复用)"""
    def __init__(self, dim: int, heads: int = 4, mlp: float = 4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, eps=1e-6)
        self.n2 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp)),
            nn.GELU(),
            nn.Linear(int(dim * mlp), dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class OceanDataEncoder(nn.Module):
    """
    时空数据编码器
    
    改造自 C-JEPA 的 FrozenSlotEncoder:
    ├─ C-JEPA: oracle slots (网格位置 → 嵌入)
    └─ 本框架: 参数特征 (传感器参数 → 时空嵌入)
    """
    def __init__(self, 
                 input_dim: int = N_CHANNELS,
                 hidden_dim: int = 128,
                 output_dim: int = SENSOR_DIM,
                 frozen: bool = False):
        super().__init__()
        
        # CNN 提取空间特征 (传感器网络的相邻性)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # 可选: 冻结 (如C-JEPA的oracle编码器)
        if frozen:
            for p in self.parameters():
                p.requires_grad_(False)
    
    def forward(self, data: torch.Tensor, missing_mask: torch.Tensor = None) -> torch.Tensor:
        """
        data: (B, T, N_CHANNELS)
        missing_mask: (B, T, N_CHANNELS) 可选
        
        输出: (B, T, N_CHANNELS, SENSOR_DIM)
        """
        B, T, C = data.shape
        
        # 展平: (B*T, N_CHANNELS)
        x = data.reshape(B*T, C)
        
        # 编码: (B*T, SENSOR_DIM)
        x = self.spatial_encoder(x)
        
        # 恢复形状: (B, T, N_CHANNELS, SENSOR_DIM)
        # 注: 这里简化处理，实际应该是 (B, T, N_PARAMS, SENSOR_DIM)
        # 为了与C-JEPA架构对齐，我们把N_CHANNELS作为"虚拟对象"
        x = x.unsqueeze(-1).expand(B, T, C, SENSOR_DIM)
        x = x.reshape(B, T, C, SENSOR_DIM)
        
        return x


class OceanDataRepairPredictor(nn.Module):
    """
    水质数据修复预测器
    
    改造自 C-JEPA 的 MaskedSlotPredictor:
    ├─ C-JEPA: mask_token + TimePE + id_proj(anchor) → 查询令牌
    └─ 本框架: mask_token + TimePE + 空间邻域信息 → 修复令牌
    """
    def __init__(self,
                 sensor_dim: int = SENSOR_DIM,
                 n_channels: int = N_CHANNELS,
                 window_len: int = WINDOW_LEN,
                 depth: int = 4,
                 heads: int = 4):
        super().__init__()
        
        self.sensor_dim = sensor_dim
        self.n_channels = n_channels
        self.window_len = window_len
        
        # 掩码令牌 (对应C-JEPA的mask_token)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, sensor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # 时间位置编码 (从C-JEPA复用)
        self.register_buffer("time_pe", sincos_1d(window_len, sensor_dim))
        
        # 空间位置编码 (新增: C-JEPA没有, 因为slot是permutation equivariant)
        # 这里用来编码传感器-参数的二维位置
        self.channel_embed = nn.Embedding(n_channels, sensor_dim)
        
        # Transformer 块 (从C-JEPA复用)
        self.blocks = nn.ModuleList(
            [Block(sensor_dim, heads) for _ in range(depth)]
        )
        
        # 输出投影
        self.norm = nn.LayerNorm(sensor_dim, eps=1e-6)
        self.to_out = nn.Linear(sensor_dim, sensor_dim)
    
    def forward(self, 
                embeddings: torch.Tensor,
                missing_mask: torch.Tensor,
                use_temporal_context: bool = True) -> torch.Tensor:
        """
        embeddings: (B, T, N_CHANNELS, SENSOR_DIM) 或 (B, T*N_CHANNELS, SENSOR_DIM)
        missing_mask: (B, T, N_CHANNELS)
        
        改造自 C-JEPA MaskedSlotPredictor.forward():
        ├─ C-JEPA: 区分 real vs query, 使用 is_q 掩码
        └─ 本框架: 同样的逻辑应用于时空维度
        """
        B, T, C, D = embeddings.shape if embeddings.dim() == 4 else (
            embeddings.shape[0], 
            embeddings.shape[1] // self.n_channels, 
            self.n_channels, 
            embeddings.shape[-1]
        )
        
        # ========== 构建查询令牌 ==========
        # real: 完整数据 + 位置编码
        real = embeddings.clone()
        real = real + self.time_pe[None, :T, None, :]              # 加时间PE
        real = real + self.channel_embed.weight[None, None, :, :]  # 加空间PE
        
        # query: 掩码令牌 + 位置编码
        query = self.mask_token.expand(B, T, C, -1)
        query = query + self.time_pe[None, :T, None, :]
        query = query + self.channel_embed.weight[None, None, :, :]
        
        # 应用掩码规则
        # 在缺失位置使用query, 在完整位置保持real
        missing_mask = missing_mask.unsqueeze(-1)  # (B, T, C, 1)
        x = torch.where(missing_mask.bool(), query, real)
        
        # ========== Transformer 编码 ==========
        # 展平为序列: (B, T*C, D)
        x = x.reshape(B, T * C, D)
        
        for blk in self.blocks:
            x = blk(x)
        
        # 输出投影
        x = self.to_out(self.norm(x))
        
        # 恢复形状: (B, T, C, D)
        return x.view(B, T, C, D)


# ============================================================================
# 第3部分: 训练循环 (改造自 C-JEPA train)
# ============================================================================

def param_groups(modules: List[nn.Module], wd: float) -> List[Dict]:
    """参数分组用于差异化权重衰减 (从C-JEPA复用)"""
    np_ = [(n, p) for m in modules for n, p in m.named_parameters() 
           if p.requires_grad]
    nd = [p for n, p in np_ if p.ndim < 2 or n.endswith("bias")]
    d = [p for n, p in np_ if p.ndim >= 2 and not n.endswith("bias")]
    return [
        {"params": d, "weight_decay": wd},
        {"params": nd, "weight_decay": 0.0}
    ]


def train_ocean_repair(
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 5e-4,
    wd: float = 0.05,
    device: str = None,
    corruption_type: str = "mixed",
    data_path: str = None
) -> Dict:
    """
    训练海洋水质修复模型
    
    改造自 C-JEPA train():
    ├─ 输入数据: (clean, corrupted, missing_mask) 替代 (video, slot_idx, labels)
    ├─ 编码器: OceanDataEncoder 替代 FrozenSlotEncoder
    ├─ 预测器: OceanDataRepairPredictor 替代 MaskedSlotPredictor
    ├─ 损失: MSE(predicted, ground_truth) 仅在缺失区域
    └─ 评估: PSNR/SSIM 替代 interaction gap
    """
    device = device or ("cuda" if torch.cuda.is_available() 
                        else "mps" if torch.backends.mps.is_available() 
                        else "cpu")
    
    print(f"\n{'='*70}")
    print(f"OCEAN WATER QUALITY DATA REPAIR TRAINING")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Corruption Type: {corruption_type}")
    print(f"Batch Size: {batch_size} | Learning Rate: {lr}")
    print(f"{'='*70}\n")
    
    # ========== 数据加载 ==========
    ds_train = OceanDataRepairDataset(
        data_path=data_path,
        corruption_type=corruption_type,
        train=True
    )
    ds_val = OceanDataRepairDataset(
        data_path=data_path,
        corruption_type=corruption_type,
        train=False
    )
    
    loader_train = DataLoader(
        ds_train, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False
    )
    
    # ========== 模型初始化 ==========
    encoder = OceanDataEncoder(frozen=False).to(device)
    predictor = OceanDataRepairPredictor().to(device)
    
    # ========== 优化器 ==========
    opt = optim.AdamW(
        param_groups([encoder, predictor], wd),
        lr=lr
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    # ========== 训练循环 ==========
    losses_repair = []
    losses_valid = []
    losses_total = []
    psnr_scores = []
    
    best_loss = float('inf')
    step = 0
    
    for epoch in range(epochs):
        # --- 训练阶段 ---
        encoder.train()
        predictor.train()
        
        for clean, corrupted, missing_mask in loader_train:
            clean = clean.to(device)         # (B, T, C)
            corrupted = corrupted.to(device)
            missing_mask = missing_mask.to(device)
            
            # 编码
            clean_embed = encoder(clean, missing_mask)          # (B, T, C, D)
            corrupted_embed = encoder(corrupted, missing_mask)  # (B, T, C, D)
            
            # 预测
            repaired_embed = predictor(corrupted_embed, missing_mask)
            
            # ========== 损失计算 ==========
            # 修复损失: 仅在缺失区域计算
            missing_mask_expanded = missing_mask.unsqueeze(-1)  # (B, T, C, 1)
            
            loss_repair = F.mse_loss(
                repaired_embed[missing_mask_expanded.bool()],
                clean_embed[missing_mask_expanded.bool()]
            )
            
            # 验证损失: 在完整区域计算 (防止过拟合)
            valid_mask = 1 - missing_mask
            if valid_mask.sum() > 0:
                valid_mask_expanded = valid_mask.unsqueeze(-1).bool()
                loss_valid = F.mse_loss(
                    repaired_embed[valid_mask_expanded],
                    clean_embed[valid_mask_expanded]
                )
            else:
                loss_valid = torch.tensor(0.0, device=device)
            
            # 总损失
            loss = loss_repair + 0.1 * loss_valid
            
            # 反向传播
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(predictor.parameters()), 
                max_norm=1.0
            )
            opt.step()
            
            # 记录指标
            losses_repair.append(loss_repair.item())
            losses_valid.append(float(loss_valid) if isinstance(loss_valid, torch.Tensor) 
                               else loss_valid)
            losses_total.append(loss.item())
            
            # 计算PSNR (诊断)
            with torch.no_grad():
                mse = F.mse_loss(repaired_embed, clean_embed)
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
                psnr_scores.append(psnr.item())
            
            # 打印日志
            if step % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Step {step:5d} | "
                      f"Loss_repair: {loss_repair.item():.4f} | "
                      f"Loss_valid: {float(loss_valid):.4f} | "
                      f"Loss_total: {loss.item():.4f} | "
                      f"PSNR: {psnr.item():.2f} dB")
            
            step += 1
        
        # --- 验证阶段 ---
        encoder.eval()
        predictor.eval()
        
        val_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for clean, corrupted, missing_mask in loader_val:
                clean = clean.to(device)
                corrupted = corrupted.to(device)
                missing_mask = missing_mask.to(device)
                
                clean_embed = encoder(clean)
                corrupted_embed = encoder(corrupted)
                repaired_embed = predictor(corrupted_embed, missing_mask)
                
                missing_mask_expanded = missing_mask.unsqueeze(-1)
                loss = F.mse_loss(
                    repaired_embed[missing_mask_expanded.bool()],
                    clean_embed[missing_mask_expanded.bool()]
                )
                
                val_loss += loss.item()
                n_batches += 1
        
        val_loss /= max(1, n_batches)
        print(f"\n  → Validation Loss: {val_loss:.4f}\n")
        
        if val_loss < best_loss:
            best_loss = val_loss
            # 保存最佳模型 (可选)
        
        scheduler.step()
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETED")
    print(f"Best Validation Loss: {best_loss:.4f}")
    print(f"Final PSNR: {psnr_scores[-1]:.2f} dB")
    print(f"{'='*70}\n")
    
    return {
        "encoder": encoder,
        "predictor": predictor,
        "losses_repair": losses_repair,
        "losses_valid": losses_valid,
        "losses_total": losses_total,
        "psnr_scores": psnr_scores,
        "loader_train": loader_train,
        "loader_val": loader_val,
        "device": device
    }


# ============================================================================
# 第4部分: 快速测试
# ============================================================================

if __name__ == "__main__":
    # 生成合成数据并训练
    out = train_ocean_repair(
        epochs=5,
        batch_size=16,
        corruption_type="mixed"
    )
    
    print("✓ Model trained successfully!")
    print(f"  - Encoder: {out['encoder']}")
    print(f"  - Predictor: {out['predictor']}")
    print(f"  - Final PSNR: {out['psnr_scores'][-1]:.2f} dB")

# AI 智能找矿 — 无监督学习 MVP 完全上手指南

> 本 README 面向零基础新手，逐行讲解项目代码的原理、目的和操作方法。读完后你将能独立运行、理解和修改整个项目。

---

## 目录

1. [项目是做什么的](#1-项目是做什么的)
2. [目录结构](#2-目录结构)
3. [环境搭建（从零开始）](#3-环境搭建从零开始)
4. [数据说明](#4-数据说明)
5. [代码逐模块讲解](#5-代码逐模块讲解)
6. [如何运行](#6-如何运行)
7. [输出文件说明](#7-输出文件说明)
8. [核心概念科普](#8-核心概念科普)
9. [常见问题 FAQ](#9-常见问题-faq)
10. [下一步可以做什么](#10-下一步可以做什么)

---

## 1. 项目是做什么的

### 一句话概括

用**人工智能（自编码器神经网络）**自动分析地表土壤的化学元素数据，找出"异常点"，再用独立的钻孔数据验证这些异常点是否真的对应地下矿藏，最终输出**找矿靶区清单**。

### 为什么用"无监督学习"

传统找矿需要大量"已知矿点"作为标签来训练模型（监督学习）。但在勘探初期，我们往往**没有标签**——不知道哪里有矿。无监督学习的思路是：

> 让模型自己学习"正常土壤的化学特征长什么样"，然后把**重建不出来**的点判定为异常（可能有矿）。

这就像一个人看了一万张普通石头的照片，再看一块矿石时会觉得"不对劲"——他不需要被人教过"这是矿石"，只需要知道"正常的石头长什么样"。

### 主线方法：自编码器（Autoencoder, AE）

自编码器是一种神经网络，它的工作是"**输入什么，就尽量输出什么**"（重建）。网络中间有一个"瓶颈层"（只有 8 个神经元），迫使模型学会用最精简的方式概括数据特征。

- **正常点**：特征符合普遍规律，模型能很好地重建 → 重建误差小
- **异常点**（可能有矿）：元素组合特殊，模型重建不好 → 重建误差大

我们把每个点的重建误差作为"异常分数"，分数越高越可能是矿化异常。

### 验证方法：钻孔数据独立验证

我们有 477 个钻孔的实际化验数据（铜、钼、金品位）。这些数据**完全不参与模型训练**，只用来验证：

> 模型找出的高异常点，附近是否真的有见矿钻孔？

用 AUC（曲线下面积）量化验证效果，AUC > 0.6 说明模型比随机猜好，链路通过。

---

## 2. 目录结构

```
D:\project\mymine\
├── data\                              # 原始数据（不要修改）
│   ├── Jiama geochemistry data 2020.8.30-iogas.xlsx   # 土壤化探数据
│   └── 数据库-川大 2025.4.29.xlsx                       # 钻孔数据
├── output\                            # 运行后生成的结果（自动创建）
│   ├── anomaly_scores.csv             # 9021个点的异常分数表
│   ├── target_areas.csv               # 靶区清单
│   ├── ae_vs_if_comparison.csv        # AE vs IF 对比指标
│   ├── 01_coordinate_verification.png # 坐标核验图
│   ├── 02_ae_training_loss.png        # AE训练损失曲线
│   ├── 03_ae_anomaly_distribution.png # 异常分数分布图
│   ├── 04_ae_vs_if_spatial.png        # AE vs IF 空间对比图
│   ├── 05_roc_and_gkf_comparison.png  # ROC曲线与交叉验证图
│   └── 06_prediction_map.png          # 最终预测图
├── ae_unsupervised_mvp.ipynb          # ★ 主程序（Jupyter Notebook）
├── run_pipeline.py                     # 从notebook提取的Python脚本（用于无界面运行）
├── build_notebook.py                   # 用于生成notebook的脚本（一般不用管）
└── README.md                           # 本文件
```

---

## 3. 环境搭建（从零开始）

### 3.1 你需要什么

- **Windows 系统**
- **Python 3.11**（项目使用的虚拟环境位于 `D:\code\language\learning\.venv\`）
- **NVIDIA GPU**（可选，没有也能用 CPU 跑，只是慢一些）

### 3.2 检查 Python 环境

打开 PowerShell，输入：

```powershell
D:\code\language\learning\.venv\Scripts\python.exe --version
```

应该显示 `Python 3.11.x`。

### 3.3 检查必要的库是否已安装

```powershell
D:\code\language\learning\.venv\Scripts\python.exe -c "import pandas, numpy, torch, sklearn, matplotlib, scipy, openpyxl; print('所有库已安装')"
```

如果报错说某个库没装，用以下命令安装：

```powershell
D:\code\language\learning\.venv\Scripts\python.exe -m pip install pandas numpy torch scikit-learn matplotlib scipy openpyxl
```

### 3.4 （可选）将虚拟环境注册为 Jupyter Kernel

如果你想在 Jupyter Notebook 界面中交互式运行代码，需要先把虚拟环境注册为 Jupyter 内核：

```powershell
D:\code\language\learning\.venv\Scripts\python.exe -m pip install ipykernel
D:\code\language\learning\.venv\Scripts\python.exe -m ipykernel install --user --name=learning-venv --display-name="learning (.venv)"
```

注册成功后，在 Jupyter 中打开 notebook，点击右上角的内核名称，选择 `learning (.venv)` 即可。

---

## 4. 数据说明

### 4.1 土壤化探数据

**文件**：`data\Jiama geochemistry data 2020.8.30-iogas.xlsx`，sheet 名 `data`

| 项目 | 说明 |
|------|------|
| 行数 | 9,021 个地表土壤采样点 |
| 列数 | 55 列 |
| 坐标列 | `East`（东向坐标，范围 374,235~384,798 米）、`North`（北向坐标，范围 3,277,714~3,292,001 米） |
| 特征列 | **49 个地球化学元素列**（10 个主量元素氧化物如 Al2O3%、CaO%，加 S 和 38 个微量元素如 Cu、Mo、Au 等，单位 ppm 或 %） |
| 弃用列 | `H`（海拔，缺失 78%）、`rock type`（岩石类型，缺失 78%） |
| 其他列 | `Analysis No`（样品编号）、`sample type`（样品类型） |

**关键数据特征**：
- Cu（铜）最大值 354,700 ppm，说明矿区铜矿化极强
- 只有 1 个 0 值、50 个 NaN 值，数据质量较好
- 坐标是 UTM 投影坐标（米），可以直接计算两点间距离

### 4.2 钻孔数据

**文件**：`data\数据库-川大 2025.4.29.xlsx`，有两个 sheet：

#### Sheet 1：`空间位置`（477 个钻孔的位置信息）

| 列名 | 含义 |
|------|------|
| `HOLE` | 钻孔编号（如 ZK002） |
| `EAST` | 东向坐标（**带 31 度带号**，如 31,378,863，需要减去 31,000,000 才能和土壤坐标对齐） |
| `NORTH` | 北向坐标 |
| `H 海拔` | 孔口海拔 |
| `TDEPTH` | 钻孔总深度 |
| `勘探线` | 勘探线编号 |

#### Sheet 2：`分析数据`（114,082 段岩心化验结果）

| 列名 | 含义 |
|------|------|
| `HOLE` | 所属钻孔编号 |
| `FROM` / `TO` | 该段岩心的起止深度（米） |
| `INT` | 段长（米） |
| `SAMPLE-ID` | 样品编号 |
| `Cu %` | 铜品位（百分比） |
| `Mo %` | 钼品位（百分比） |
| `Au ppm` | 金品位（克/吨） |
| `Ag ppm` / `Pb %` / `Zn %` | 其他元素 |
| `岩性` | 岩石类型 |

> **注意**：`Cu %`、`Mo %` 等列中可能含有字符串（如 `<0.001` 表示低于检出限），代码中会自动转换为数值，无法转换的变为空值。

---

## 5. 代码逐模块讲解

Notebook 共 8 个步骤，下面逐一讲解。

---

### Step 0：导入与全局配置

**做什么**：导入所有需要的 Python 库，设置随机种子、计算设备、文件路径等全局参数。

**关键代码解释**：

```python
import torch
import torch.nn as nn
```
PyTorch 是构建和训练神经网络的库。`nn` 模块包含神经网络的各种层（全连接层、激活函数等）。

```python
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
自动选择计算设备：有 NVIDIA GPU 就用 GPU（快），没有就用 CPU（慢但能跑）。

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
```
设置随机种子。神经网络训练中有很多随机操作（参数初始化、数据打乱等），固定种子可以保证**每次运行结果完全一样**，方便复现和调试。

```python
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
```
设置中文字体，让图表中的中文能正常显示，负号不会变成方块。

---

### Step 1：读取数据

**做什么**：用 pandas 读取两个 Excel 文件。

**关键代码解释**：

```python
df_soil = pd.read_excel(SOIL_PATH, sheet_name='data')
```
读取土壤数据的 `data` 工作表。`pd.read_excel` 会自动解析 Excel 文件，返回一个 DataFrame（类似表格的数据结构）。

```python
df_soil.columns = df_soil.columns.str.strip()
```
去除列名首尾的空格。原始数据中有些列名带空格（如 `'Y ppm '`），不处理的话后续按列名访问会报错。

```python
df_pos = pd.read_excel(DRILL_PATH, sheet_name='空间位置')
df_ana = pd.read_excel(DRILL_PATH, sheet_name='分析数据')
```
分别读取钻孔数据的两个工作表。

**输出**：打印数据的形状（行数×列数）、列名、坐标范围、Cu 最大值、缺失率等，帮助你确认数据读取正确。

---

### Step 2：坐标核验（先行步骤）

**做什么**：确认土壤点和钻孔在同一个坐标系统、同一个地理区域。

**为什么重要**：如果坐标不对齐，后续的"异常点附近是否有矿"的验证就毫无意义。这是找矿项目的**红线检查**。

**关键代码解释**：

```python
df_pos['EAST_corr'] = df_pos['EAST'] - 31_000_000
```
钻孔的 EAST 坐标带了 31 度带号（如 31,378,863），减去 31,000,000 后得到 378,863，才能和土壤坐标（374k~385k）比较。这是高斯-克吕格/UTM 投影的标准处理。

```python
east_overlap = (df_pos['EAST_corr'].min() < df_soil['East'].max()) and ...
```
检查两组数据的坐标范围是否有重叠。有重叠说明在同一个区域。

**输出**：一张散点图，土壤点用颜色表示 Cu 含量，钻孔用红色三角标记。可以直观看到两者空间分布是否一致。

---

### Step 3：特征工程

**做什么**：把原始的 49 个元素数据处理成神经网络能接受的格式。

**处理流程（4 步）**：

#### 3.1 选择特征列

```python
start_idx = all_cols.index('sample type') + 1
end_idx = all_cols.index('H')
feature_cols = all_cols[start_idx:end_idx]
```
自动定位 49 个元素列：`sample type` 之后、`H` 之前的所有列。这样写的好处是不用手动列 49 个列名，也不容易漏。

#### 3.2 0 → NaN

```python
X = X_raw.replace(0, np.nan)
```
在地球化学数据中，0 通常表示"低于检出限"而不是真的为 0。把 0 替换为空值（NaN），后续统一处理。

#### 3.3 log1p 变换

```python
X_log = np.log1p(X)
```
对每个元素取 `log(1 + x)`。**为什么要做这一步？**

地球化学数据的分布极度偏斜——大部分点 Cu 含量只有几十 ppm，但矿化点可以达到几十万 ppm。直接用原始值，大值会"压制"小值，模型只关注那些极端大值。取对数后：
- 数据分布更接近正态分布
- 压缩了极端值的影响
- 保留了相对大小关系（log 是单调函数）

`log1p` 而不是 `log`，是因为 `log(0)` 无意义，而 `log1p(0) = log(1) = 0`。

#### 3.4 中位数填充 + 标准化

```python
medians = X_log.median()
X_log_filled = X_log.fillna(medians)
```
空值用该列的中位数填充。用中位数而不是平均值，是因为中位数对极端值不敏感（更稳健）。

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log_filled)
```
标准化：每个元素列减去均值、除以标准差，使所有列的均值为 0、标准差为 1。

**为什么必须标准化？** 不同元素的数值范围差异巨大（Al2O3 约 13%，Au 约 0.05 ppm）。如果不标准化，数值大的元素在 MSE 损失中会占主导地位，模型等于只看那几个大值元素。标准化后每个元素对损失的贡献均等。

**最终输出**：`X_scaled` 是一个 9021×49 的 numpy 数组，每行是一个采样点，每列是一个处理后的元素特征。

---

### Step 4：自编码器（AE）训练

**做什么**：定义神经网络结构，用土壤数据训练，让模型学会"重建"正常的地球化学特征。

#### 4.1 网络结构

```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim=49, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim),
        )
```

网络结构是**对称的沙漏形**：

```
输入(49) → 32 → 16 → 8(瓶颈) → 16 → 32 → 输出(49)
    \________编码器________/      \________解码器________/
```

- **编码器**：把 49 维的输入逐步压缩到 8 维（瓶颈层）
- **解码器**：把 8 维的压缩表示逐步还原回 49 维
- **ReLU 激活函数**：在每层之间加入非线性，让网络能学习复杂的模式。公式是 `f(x) = max(0, x)`，即负数变 0，正数不变
- **输出层无激活函数**：因为我们要重建连续的数值（标准化后的元素含量），直接输出原始值即可

**为什么网络这么小？** 总参数只有 4,569 个。数据只有 9,021 个点，如果网络太大（比如几百层、几万个参数），模型会"背答案"（过拟合）——把每个点的特征都记下来，重建误差都很小，反而无法区分异常点。小网络迫使模型学习**普遍规律**，而不是记住每个点。

#### 4.2 训练过程

```python
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```
- **MSE 损失**（均方误差）：衡量输入和输出之间的差异，公式是 `mean((输入 - 输出)²)`。重建得越好，MSE 越小
- **Adam 优化器**：一种自适应学习率的优化算法，是训练神经网络最常用的优化器
- **学习率 lr=1e-3**（0.001）：控制每次参数更新的步长。太大会震荡不收敛，太小会训练极慢。1e-3 是 Adam 的常用默认值

```python
dataset = TensorDataset(torch.FloatTensor(X_scaled))
loader = DataLoader(dataset, batch_size=128, shuffle=True)
```
- **batch_size=128**：每次训练用 128 个样本计算一次梯度、更新一次参数。不用全部 9021 个点一起算（太占内存），也不用一个一个算（太慢），128 是平衡选择
- **shuffle=True**：每个 epoch 打乱数据顺序，防止模型学到数据顺序的规律

#### 4.3 早停（Early Stopping）

```python
if avg_loss < best_loss - 1e-6:
    best_loss = avg_loss
    best_state = ...  # 保存当前最优模型参数
    counter = 0
else:
    counter += 1
    if counter >= patience:  # patience=15
        print(f'早停于 epoch {epoch+1}')
        break
```

早停是防止过拟合的关键技术：
- 记录训练过程中损失最小的那组模型参数
- 如果连续 15 个 epoch（patience=15）损失都没有明显下降，就停止训练
- 最后恢复到损失最小的那组参数

这样做的原因是：训练初期模型在学习普遍规律（损失下降），训练后期模型开始"背答案"（过拟合，训练损失还在降但泛化能力变差）。早停在过拟合发生前停下来。

> 本次运行中训练了全部 300 个 epoch 没有触发早停，说明模型还在持续学习，没有过拟合（小网络的优势）。

---

### Step 5：异常评分

**做什么**：用训练好的 AE 对每个点计算重建误差，归一化为异常分数。

```python
def compute_recon_mse(model, X_data, device=DEVICE, batch_size=512):
    model.eval()
    mse_list = []
    with torch.no_grad():
        for i in range(0, len(X_data), batch_size):
            batch = torch.FloatTensor(X_data[i:i+batch_size]).to(device)
            recon = model(batch).cpu().numpy()
            mse = np.mean((X_data[i:i+batch_size] - recon) ** 2, axis=1)
            mse_list.append(mse)
    return np.concatenate(mse_list)
```

- `model.eval()`：切换到评估模式，关闭训练时才用的机制（如 Dropout）
- `torch.no_grad()`：不计算梯度，节省内存和时间（推理阶段不需要反向传播）
- 对每个点，计算原始特征和重建特征之间的 MSE

```python
anomaly_score_ae = (recon_mse - mse_min) / (mse_max - mse_min)
```
把 MSE 归一化到 [0, 1] 区间，方便解释和比较。分数越接近 1，越异常。

**输出**：
- 异常分数的分布统计（5%、50%、95%、99% 分位数）
- 直方图：看分数分布是否合理
- 空间散点图：看高异常点在空间上的分布

---

### Step 6：IsolationForest 基线对比

**做什么**：用经典的孤立森林算法做同样的异常检测，作为对比基线。

**什么是孤立森林（IsolationForest）？**

这是一种基于决策树的经典异常检测算法。核心思想：

> 异常点因为特征特殊，更容易被"孤立"（用更少的随机划分就能单独分出来）。

算法随机选择特征和阈值构建多棵决策树，一个点在所有树中的平均路径长度越短，越可能是异常点。

```python
iso_forest = IsolationForest(
    n_estimators=200,      # 构建 200 棵树
    contamination=0.05,     # 预期异常点比例 5%
    random_state=SEED,
    n_jobs=-1               # 用所有 CPU 核心并行
)
iso_forest.fit(X_scaled)
if_scores = -iso_forest.score_samples(X_scaled)
```

- `contamination=0.05`：告诉算法大约 5% 的点是异常的，用于设定判定阈值
- `score_samples` 返回的是负值（越小越异常），取负后变成越大越异常，和 AE 分数方向一致

**为什么需要基线？** 单独看 AE 的 AUC=0.65 不知道好不好。和经典算法对比后，才能说"AE 比传统方法强/弱/相当"。这是科学研究的基本规范。

---

### Step 7：钻孔独立验证（核心验证环节）

> **红线**：钻孔数据从这里才第一次出现，**绝对没有**进入前面的特征矩阵或模型训练。这保证了验证的独立性——模型从来没"见过"钻孔数据，验证结果才可信。

#### 7.1 CuEq 计算与矿化段圈定

```python
for col in ['Cu %', 'Mo %', 'Au ppm', ...]:
    df_ana[col] = pd.to_numeric(df_ana[col], errors='coerce')
```
先把化学成分列转为数值。原始数据中可能有字符串（如 `<0.001`），`errors='coerce'` 会把无法转换的值变成空值（NaN）。

```python
df_ana['CuEq'] = df_ana['Cu %'] + 8.33 * df_ana['Mo %']
```
计算**铜当量（CuEq）**：把钼折算成等价的铜。

- 为什么系数是 8.33？这是基于铜钼价格比的经验折算系数（钼价约为铜价的 8.33 倍时，1% Mo 等价于 8.33% Cu）
- 为什么要折算？甲玛是铜钼多金属矿，单独看 Cu 会漏掉钼矿化的段，CuEq 综合考虑两种金属

```python
mineralized = df_ana[df_ana['CuEq'] >= 0.3]
```
CuEq ≥ 0.3% 的段圈定为**矿化段**。0.3% 是铜钼矿的常用工业品位阈值。

#### 7.2 见矿孔识别

```python
ore_holes = mineralized['HOLE'].unique()
ore_pos = df_pos[df_pos['HOLE'].isin(ore_holes)]
```
- 只要一个钻孔中有至少一段矿化段，就称为**见矿孔**
- 本次数据中 462/477 孔见矿（96.9%），说明甲玛矿区整体矿化程度很高

#### 7.3 空间关联

```python
ore_tree = cKDTree(ore_coords)
dist_to_ore, _ = ore_tree.query(coords, k=1)
```
用 **KD-Tree** 高效计算每个土壤点到最近见矿孔的距离。KD-Tree 是一种空间索引数据结构，比暴力计算（9021 × 462 次距离计算）快得多。

```python
HIT_RADIUS = 500  # 米
labels = (dist_to_ore <= HIT_RADIUS).astype(int)
```
- 距见矿孔 ≤ 500 米的土壤点标记为**正样本**（标签=1，表示附近有矿）
- 500 米是找矿中的常用关联半径（土壤异常的空间影响范围）
- 本次 20.74% 的点是正样本

#### 7.4 AUC 与命中率

```python
auc = roc_auc_score(labels, scores)
```
**AUC（曲线下面积）**是衡量二分类模型好坏的核心指标：

- AUC = 1.0：完美预测
- AUC = 0.5：和随机猜一样
- AUC > 0.6：模型有预测能力（本项目的通过线）

AUC 的直观含义：**随机选一个正样本和一个负样本，模型给正样本打更高分的概率**。

```python
threshold = np.percentile(scores, 100 - top_percent)  # Top-5% 阈值
top_mask = scores >= threshold
hit_rate = labels[top_mask].mean()
```
**Top-5% 命中率**：模型认为最异常的 5% 点中，有多少比例真的在见矿孔 500 米内。

- 基线命中率 = 20.74%（随机选点的命中率）
- AE Top-5% 命中率 = 30.31%（提升 1.46 倍）
- 说明模型选的高异常点确实比随机点更可能靠近矿

#### 7.5 GroupKFold 空间交叉验证

**为什么需要交叉验证？** 只算一次 AUC 可能有偶然性（刚好选的测试集容易/难）。交叉验证把数据分成多份，轮流做测试集，取平均结果更可靠。

**为什么用 GroupKFold 而不是普通 KFold？** 普通 KFold 随机划分数据，但空间上相邻的点特征高度相关（空间自相关）。如果训练集和测试集有相邻点，模型相当于"间接见过"测试点附近的数据，导致 AUC 虚高（**空间泄漏**）。

```python
GRID_SIZE = 1000  # 米
grid_x = (coords[:, 0] // GRID_SIZE).astype(int)
grid_y = (coords[:, 1] // GRID_SIZE).astype(int)
groups = grid_x * 100000 + grid_y
```
把研究区划分成 1km × 1km 的网格，每个网格是一个"组"。GroupKFold 保证**同一个网格的点不会同时出现在训练集和测试集**，从根本上防止空间泄漏。

```python
gkf = GroupKFold(n_splits=5)
for fold, (train_idx, test_idx) in enumerate(gkf.split(scores, labels, groups)):
    auc = roc_auc_score(labels[test_idx], scores[test_idx])
```
5 折交叉验证：每次用 4 份数据的网格做训练、1 份做测试，轮流 5 次，报告平均 AUC 和标准差。

- AE GroupKFold AUC = 0.6400 ± 0.055（稳定）
- IF GroupKFold AUC = 0.6033 ± 0.119（波动大，某些折接近随机）

标准差越小说明模型在不同空间区域表现越一致，越可靠。

---

### Step 8：交付物生成

#### 8.1 异常分数表

```python
df_scores = pd.DataFrame({
    'Analysis No': df_soil['Analysis No'].values,
    'East': coords[:, 0],
    'North': coords[:, 1],
    'AE_anomaly_score': anomaly_score_ae,
    'IF_anomaly_score': anomaly_score_if,
    'dist_to_ore_m': dist_to_ore,
    'within_500m_ore': labels,
})
```
每个采样点一行，包含坐标、AE/IF 异常分数、到最近见矿孔的距离、是否在 500 米内。按 AE 分数降序排列，最异常的点在最前面。

#### 8.2 靶区清单（DBSCAN 空间聚类）

```python
TOP_PCT = 5
threshold = np.percentile(anomaly_score_ae, 100 - TOP_PCT)
top_mask = anomaly_score_ae >= threshold
top_coords = coords[top_mask]
```
先取 AE 异常分数最高的 5% 点（452 个点）作为候选异常点。

```python
db = DBSCAN(eps=500, min_samples=5)
cluster_labels = db.fit_predict(top_coords)
```
用 **DBSCAN 聚类算法**把空间上聚集的高异常点圈成靶区：
- `eps=500`：两个点距离 ≤ 500 米就算相邻
- `min_samples=5`：一个簇至少要有 5 个点（少于 5 个的孤立点标记为噪声）

DBSCAN 的优点是不需要预先指定簇的数量，能自动发现任意形状的簇，还能识别噪声点。

```python
for cid in range(n_clusters):
    center_east = c_coords[:, 0].mean()
    area = (x_max - x_min) * (y_max - y_min) / 1e6  # km²
```
对每个簇（靶区）计算：中心点坐标、包含的异常点数、平均/最高/最低异常分数、包围盒面积。

本次识别出 3 个靶区：
- **T01**（主靶区）：403 个点，覆盖矿区主矿化带
- **T03**：9 个点，东北部小靶区
- **T02**：7 个点，东部小靶区

#### 8.3 预测图

在一张图上叠加：
- 所有土壤点（浅灰色背景）
- AE 异常分数（黄→红渐变，只显示 Top-5%）
- 见矿孔（红色三角）
- 非见矿孔（灰色倒三角）
- 靶区范围（蓝色虚线框 + 编号）

这是项目最核心的可视化成果，一眼就能看出异常区和已知矿点的对应关系。

#### 8.4 AE-vs-IF 对比指标表

把所有验证指标汇总成一张表，方便写报告时直接引用。

---

## 6. 如何运行

### 方式一：直接运行 Python 脚本（最简单，推荐新手）

不需要装 Jupyter，直接在 PowerShell 中运行：

```powershell
cd D:\project\mymine
D:\code\language\learning\.venv\Scripts\python.exe run_pipeline.py
```

运行时间约 2~5 分钟（取决于电脑性能）。运行结束后，所有结果会出现在 `output\` 文件夹中。

### 方式二：Jupyter Notebook 交互式运行

适合想逐步看结果、调参数的用户。

1. 先注册虚拟环境为 Jupyter 内核（只需做一次）：
   ```powershell
   D:\code\language\learning\.venv\Scripts\python.exe -m pip install ipykernel
   D:\code\language\learning\.venv\Scripts\python.exe -m ipykernel install --user --name=learning-venv --display-name="learning (.venv)"
   ```

2. 启动 Jupyter：
   ```powershell
   D:\code\language\learning\.venv\Scripts\python.exe -m jupyter notebook
   ```

3. 在浏览器中打开 `ae_unsupervised_mvp.ipynb`

4. 点击右上角内核名称，选择 `learning (.venv)`

5. 按 `Shift + Enter` 逐个单元格运行，或点击菜单 `Cell → Run All` 一次性运行

---

## 7. 输出文件说明

| 文件名 | 类型 | 说明 |
|--------|------|------|
| `anomaly_scores.csv` | 数据表 | 9,021 个采样点的完整异常分数表，含坐标、AE/IF分数、距见矿孔距离。用 Excel 打开即可查看 |
| `target_areas.csv` | 数据表 | 3 个靶区的详细信息（编号、中心坐标、点数、分数、面积、优先级） |
| `ae_vs_if_comparison.csv` | 数据表 | AE 与 IsolationForest 的各项指标对比 |
| `01_coordinate_verification.png` | 图片 | 坐标核验图，确认土壤点和钻孔在同一区域 |
| `02_ae_training_loss.png` | 图片 | AE 训练损失曲线，看模型是否收敛 |
| `03_ae_anomaly_distribution.png` | 图片 | AE 异常分数的直方图和空间分布图 |
| `04_ae_vs_if_spatial.png` | 图片 | AE 和 IF 异常分数的空间分布并排对比 |
| `05_roc_and_gkf_comparison.png` | 图片 | ROC 曲线 + GroupKFold 交叉验证 AUC 柱状图 |
| `06_prediction_map.png` | 图片 | **最终预测图**，AE 异常 + 见矿孔 + 靶区圈定，最核心的成果图 |

---

## 8. 核心概念科普

### 自编码器（Autoencoder）

一种无监督神经网络，目标是"输入什么输出什么"。通过中间的瓶颈层压缩信息，迫使模型学习数据的本质规律。重建误差大的点就是偏离普遍规律的异常点。

### 异常检测（Anomaly Detection）

在数据中找出"不正常"的样本。在找矿中，异常的地球化学元素组合可能指示地下矿化。无监督异常检测不需要已知矿点标签，适合勘探初期。

### AUC（Area Under Curve）

衡量二分类模型性能的指标。AUC=0.5 等于随机猜，AUC=1.0 是完美分类。在本项目中，AUC 衡量"异常分数高的点是否真的靠近见矿孔"。

### MSE（均方误差）

`mean((真实值 - 预测值)²)`。衡量模型重建的好坏，值越小重建越好。在 AE 中用作损失函数和异常分数的基础。

### log1p 变换

`log(1 + x)`。对极度偏斜的地球化学数据做对数变换，压缩极端值，使分布更均匀。

### 标准化（StandardScaler）

`(x - 均值) / 标准差`。让所有特征的均值为 0、标准差为 1，消除不同元素量纲差异的影响。

### GroupKFold

一种交叉验证方法，按"组"划分数据，保证同一组的样本不会同时出现在训练集和测试集。在空间数据中，按网格分组可以防止邻近点泄漏导致的指标虚高。

### DBSCAN 聚类

基于密度的空间聚类算法。把距离近（eps 以内）且数量足够（min_samples 以上）的点聚成一类，能自动发现任意形状的簇并识别噪声点。

### CuEq（铜当量）

`Cu% + 8.33 × Mo%`。把钼折算成等价铜的综合品位指标，用于铜钼多金属矿的矿化评价。

---

## 9. 常见问题 FAQ

### Q1：运行时报错 `ModuleNotFoundError: No module named 'xxx'`

A：某个 Python 库没装。用以下命令安装：
```powershell
D:\code\language\learning\.venv\Scripts\python.exe -m pip install xxx
```

### Q2：图表中的中文显示为方块

A：字体问题。代码中已经设置了 `SimHei` 和 `Microsoft YaHei`，如果你的系统没有这两种字体，可以：
1. 安装中文字体
2. 或把代码中的字体名改成你系统有的中文字体（如 `'KaiTi'`、`'FangSong'`）

### Q3：GPU 没用上，训练很慢

A：检查是否安装了 CUDA 版本的 PyTorch。运行以下代码查看：
```python
import torch
print(torch.cuda.is_available())  # 应该输出 True
```
如果输出 False，需要安装 CUDA 版 PyTorch，或者用 CPU 跑（也能跑，只是慢一些，本项目数据量小，CPU 也能在几分钟内完成）。

### Q4：AUC 太低怎么办？

A：可以尝试以下改进：
1. **调整特征**：尝试只选用成矿相关元素（Cu、Mo、Au、Ag、Pb、Zn 等），去掉噪声元素
2. **调整网络**：尝试不同的瓶颈层维度（4、8、16）、不同层数
3. **调整训练**：增加 epoch 数、调整学习率、加入权重衰减（weight decay）
4. **特征组合**：尝试元素比值（如 Cu/Mo、Zn/Pb）作为新特征
5. **验证参数**：调整矿化阈值（0.3% → 0.5%）、关联半径（500m → 300m/1000m）

### Q5：为什么 96.9% 的钻孔都见矿？这样验证还有意义吗？

A：甲玛是已探明的大型铜钼矿，矿区内钻孔见矿率高是正常的。验证的意义不在于"区分有矿无矿"，而在于"**异常分数高的点是否更靠近高品位矿化中心**"。即使所有孔都见矿，不同孔的矿化强度差异很大，AUC 仍然能反映模型的排序能力。如果想更严格，可以用"高品位孔"（如 CuEq ≥ 1%）作为正样本。

### Q6：怎么修改参数？

A：所有关键参数都在 notebook 中有明确的变量定义，可以直接修改：
- `SEED = 42`：随机种子
- `HIT_RADIUS = 500`：见矿关联半径（米）
- `TOP_PCT = 5`：靶区候选点比例（%）
- `GRID_SIZE = 1000`：GroupKFold 网格大小（米）
- AE 训练参数：`epochs=300, batch_size=128, lr=1e-3, patience=15`
- AE 结构：`input_dim=49, latent_dim=8`（在 `Autoencoder` 类中）
- DBSCAN 参数：`eps=500, min_samples=5`

修改后重新运行对应单元格即可。

### Q7：输出文件在哪里？

A：所有输出在 `D:\project\mymine\output\` 文件夹中。CSV 文件用 Excel 打开，PNG 图片用任意图片查看器打开。

---

## 10. 下一步可以做什么

### 入门级（直接改参数）
1. 把 `latent_dim` 从 8 改成 4 或 16，看 AUC 怎么变
2. 把 `HIT_RADIUS` 从 500 改成 300 或 1000，看命中率怎么变
3. 把矿化阈值从 0.3% 改成 0.5% 或 1.0%，看验证结果

### 进阶级（改代码逻辑）
1. **特征选择**：只选用 Cu、Mo、Au、Ag、Pb、Zn、As、Sb 等成矿指示元素，去掉主量元素，看效果
2. **元素比值**：新增 Cu/Mo、Zn/Pb、(Cu+Mo)/(Pb+Zn) 等比值特征
3. **多模型集成**：把 AE 分数和 IF 分数取平均或加权，看是否提升
4. **空间插值**：用克里金插值把异常分数生成连续等值线图

### 高级（拓展研究）
1. **变分自编码器（VAE）**：在 AE 基础上加入概率建模，输出异常的置信区间
2. **深度 SVDD**：另一种深度无监督异常检测方法，学习最小包围超球
3. **多尺度验证**：用不同深度的钻孔数据（浅部/深部）分别验证
4. **时序分析**：如果有多期化探数据，做时序异常检测

---

## 附：快速验证清单

运行完成后，检查以下内容确认一切正常：

- [ ] `output\` 文件夹中有 9 个文件（3 CSV + 6 PNG）
- [ ] `anomaly_scores.csv` 有 9,021 行
- [ ] `target_areas.csv` 有 3 行（3 个靶区）
- [ ] 控制台输出中 AE 全量 AUC > 0.6
- [ ] 控制台输出中 AE GroupKFold AUC > 0.6
- [ ] `06_prediction_map.png` 中红色高异常区与红色三角（见矿孔）有重叠
- [ ] 没有报错信息

如果全部通过，恭喜你！整个无监督找矿流水线已经成功运行。

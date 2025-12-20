# mfstock: 多频率 Transformer 股票收益率预测系统

## 1. 项目简介
本项目是一个基于深度学习（Transformer）的多频率股票收益率预测框架。它能够同时处理不同频率（如月度、周度）的特征数据，通过特征融合技术预测未来收益率，并提供完整的滚动窗口训练、预测及回测流程。

## 2. 项目文件结构
```text
.
├── main.py                 # 项目入口（占位）
├── pyproject.toml          # 项目配置文件（uv 管理）
├── configs/                # 配置文件目录
│   ├── project_config.yaml # 项目基础配置
│   ├── training_config.yaml# 训练与模型配置
│   └── backtest_config.yaml# 回测配置
├── data/                   # 数据目录
│   ├── feature/            # 处理后的特征数据 (.parquet)
│   ├── target/             # 预测目标数据 (.parquet)
│   └── backtest_data/      # 回测所需基础数据
├── src/mfstock/            # 核心源代码
│   ├── dataset/            # 数据处理、Dataset、DataLoader、滚动窗口逻辑
│   ├── models/             # Transformer 模型架构、训练引擎、模型保存
│   ├── backtest/           # 回测核心逻辑
│   ├── run/                # 运行脚本 (train, backtest)
│   └── utils/              # 工具函数
├── output/                 # 训练输出（模型权重、预测结果、配置备份）
├── backtest_result/        # 回测结果输出（净值曲线、指标报告）
└── test/                   # 测试脚本
```

## 3. 项目部署
本项目使用 [uv](https://github.com/astral-sh/uv) 进行包管理和环境隔离。

### 安装 uv
请参考 [uv 官方文档](https://github.com/astral-sh/uv) 进行安装。

### 初始化环境
在项目根目录下运行以下命令以安装依赖并创建虚拟环境：
```bash
uv sync --no-dev
```

## 4. 配置文件说明
项目主要通过 `configs/` 下的 YAML 文件进行配置：

- **`project_config.yaml`**: 设置项目根目录 `project_root`。
- **`training_config.yaml`**: 模型训练的配置文件
    - `paths`: 指定数据输入和结果输出的相对路径。
    - `dataset`: 配置特征和目标文件路径、时间/股票列名、预处理方式（中位数填充、MAD 去极值、Z-Score 归一化）。
    - `dataloader`: 配置 Batch Size 和数据加载线程数。
    - `rolling_window`: 定义滚动窗口逻辑，包括各频率的 Lookback 长度、训练/验证/测试集窗口长度及调仓频率。
    - `model`: 配置 Transformer 的超参数（层数、头数、维度）及多频率融合方式。
    - `training`: 配置优化器参数、Epochs、早停（Early Stopping）策略。
- **`backtest_config.yaml`**: 回测的配置文件
    - `paths_config`: 配置回测所需的行情数据（复权价、市值、股票池、停牌限价）及待回测的因子路径。
    - `backtest_config`: 配置回测区间、中性化方式及回测频率。

## 5. 项目使用方式
项目通过 `pyproject.toml` 定义了快捷命令，使用 `uv run` 即可调用：

### 5.1 模型训练与预测
执行滚动窗口训练流程，并生成全样本预测因子：
```bash
uv run mf-train
```
训练完成后，模型权重和预测结果将保存在 `output/{experiment_id}/` 目录下。

### 5.2 因子回测
对生成的预测因子进行分层回测：
```bash
uv run mf-backtest
```
回测结果（CSV 报表和可视化图表）将保存在 `backtest_result/` 目录下。

## 6. 数据与输出结构

### Data 目录结构
- `data/feature/`: 存放 Parquet 格式的特征文件，如 `feature_monthly.parquet`。
- `data/target/`: 存放 Parquet 格式的预测目标文件。
- `data/backtest_data/`: 存放回测必备的行情数据：
    - `adj_close.parquet`: 复权收盘价。
    - `market_value.parquet`: 个股市值。
    - `stock_pool.parquet`: 股票池定义。
    - `suspension_limit.parquet`: 停牌及涨跌停信息。

### Output 目录结构
每次训练任务会生成一个唯一的 `experiment_id` 文件夹：
- `output/{experiment_id}/config.yaml`: 本次任务的配置备份。
- `output/{experiment_id}/window_{n}/best_model.pt`: 各滚动窗口训练出的最优模型。
- `output/{experiment_id}/pred_{experiment_id}.parquet`: 模型生成的全样本预测因子文件。

### Backtest Result 目录结构
- `backtest_result/pred_{experiment_id}/`:
    - `..._分组净值.csv`: 各分层组别的累计净值。
    - `..._分组回测结果.csv`: 包含年化收益、夏普比率、最大回撤等核心指标。
    - `..._分组收益率.csv`: 各组别的周期收益率明细。

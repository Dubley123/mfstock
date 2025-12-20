"""单因子回测工具。

本模块实现单因子的分组回测流程，包括：数据预处理（股票池筛选、标准化、市值中性化）、
带换手成本处理的分组回测，以及回测结果分析。文档字符串统一采用 Google 风格。
"""
# 标准库
import os
import warnings
warnings.simplefilter(action="ignore")
from pathlib import Path
from dateutil import parser

# 第三方库
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from tqdm import tqdm

# 本地模块
from mfstock.utils.misc import (
    calculate_runtime,
    normalize_frequency
)
from mfstock.utils.format_display import (
    format_multiline_block,
    format_single_line_info,
    pretty_print_df
)

__all__ = [
    "run_bt",
]


class back_test:
    """单因子回测器。

    执行因子的基础处理、分组回测和回测结果分析。
    """
    def __init__(
        self,
        factor,
        price,
        fac_name,
        dir,
        market_index,
        suspension_limit,
        start_date='2010-01-01',
        end_date='2025-06-30'
    ):
        """初始化回测器。

        Args:
            factor (pd.DataFrame): 因子数据，包含列 ['TradingDate', 'Stkcd', fac_name]。
            price (pd.DataFrame): 收盘价数据，包含 ['TradingDate', 'Stkcd', 'price']。
            fac_name (str): 因子列名。
            dir (str): 结果保存目录。
            market_index (str): 市场指数名称，用于对比与超额收益计算。
            suspension_limit (pd.DataFrame): 交易限制表，包含 ['TradingDate','Stkcd','Suspension','LimitStatus']。
            start_date (str, optional): 起始日期 YYYY-MM-DD，默认值 '2010-01-01'。
            end_date (str, optional): 结束日期 YYYY-MM-DD，默认值 '2025-06-30'。
        """
        # 转换因子列类型为 float
        factor = factor.copy()
        price = price.copy()
        suspension_limit = suspension_limit.copy()

        factor[fac_name] = factor[fac_name].astype(float)

        # 如果存在 'Accper' 列，则重命名为 TradingDate
        if 'Accper' in factor.columns:
            factor = factor.rename(columns={'Accper': 'TradingDate'})

        factor['TradingDate'] = pd.to_datetime(factor['TradingDate'])
        factor = factor[(factor['TradingDate'] >= start_date) & (factor['TradingDate'] <= end_date)]

        # 处理价格数据
        price['TradingDate'] = pd.to_datetime(price['TradingDate'])
        price = price[(price['TradingDate'] >= start_date) & (price['TradingDate'] <= end_date)]

        # 处理停牌和涨跌停状态
        suspension_limit['TradingDate'] = pd.to_datetime(suspension_limit['TradingDate'])
        # suspension_limit = suspension_limit[(suspension_limit['TradingDate'] >= start_date) & (suspension_limit['TradingDate'] <= end_date)]
        suspension_limit[['Suspension', 'LimitStatus']] = suspension_limit.groupby('Stkcd')[['Suspension', 'LimitStatus']].shift(-1)  # 对齐为下一日的交易限制状态

        # 保存实例属性
        self.start_date = start_date            # 回测开始日期
        self.end_date = end_date                # 回测结束日期
        self.price = price                      # 收盘价数据
        self.fac_name = fac_name                # 测试单因子名称
        self.factor = factor                    # 因子值
        self.dir = dir                          # 结果保存目录
        self.suspension_limit = suspension_limit  # 股票交易限制状态（1：受限，0：正常）
        self.neutralize = None                  # 因子中性化方式
        self.market_index_name = market_index   # 市场指数名称

    def __str__(self):
        lines = [
            f"Start Date: {self.start_date}",
            f"End Date: {self.end_date}",
            f"Factor Name: {self.fac_name}",
            f"Neutralize: {self.neutralize}",
            f"Market Index Name: {self.market_index_name}"
        ]
        return format_multiline_block("Backtest Configuration", lines)

    @calculate_runtime
    def factor_process(self, stock_pool, market_value=None, neutralize='cap'):
        """处理因子值：股票池筛选、标准化与市值中性化。

        Args:
            stock_pool (pd.DataFrame): 股票池矩阵，index 为日期，列为股票代码（或含小数点的代码）。
            market_value (pd.DataFrame | None, optional): 含 ['TradingDate','Stkcd','MarketValue'] 的市值数据；当
                ``neutralize='cap'`` 时用于中性化，默认值 None。
            neutralize (str, optional): 中性化方式，'cap' 表示按对数市值回归取残差，默认值 'cap'。

        Returns:
            None: 该方法就地更新实例的 ``factor``、``price``、``suspension_limit``。
        """

        # ---------------- Step 0: 股票池筛选 ----------------
        stock_pool = stock_pool.copy()
        stock_pool = stock_pool.loc[self.start_date:self.end_date]
        stock_pool.columns = [int(col.split('.')[0]) for col in stock_pool.columns]
        stock_pool.index = pd.to_datetime(stock_pool.index)

        # ---------------- Step 0.5: 排除创业板、科创板股票 ----------------
        columns_to_zero = [
            col for col in stock_pool.columns
            if 300000 <= col <= 301999 or 688000 <= col <= 689999
        ]
        stock_pool.loc[:, columns_to_zero] = 0  

        # 仅保留股票池内的股票
        stock_pool = stock_pool.stack().reset_index()
        stock_pool.columns = ['TradingDate', 'Stkcd', 'in_pool']

        merged_df = pd.merge(self.factor, stock_pool, on=['TradingDate', 'Stkcd'], how='left')
        self.factor = merged_df[merged_df['in_pool'] == 1].drop(columns='in_pool')

        merged_df = pd.merge(self.suspension_limit, stock_pool, on=['TradingDate', 'Stkcd'], how='left')
        self.suspension_limit = merged_df[merged_df['in_pool'] == 1].drop(columns='in_pool')

        merged_df = pd.merge(self.price, stock_pool, on=['TradingDate', 'Stkcd'], how='left')
        self.price = merged_df[merged_df['in_pool'] == 1].drop(columns='in_pool')

        # ---------------- Step 1: 因子基本处理 ----------------
        # 删除 NaN 和 inf 值
        self.factor[self.fac_name] = self.factor[self.fac_name].replace([np.inf, -np.inf], np.nan).dropna()

        # ---------------- Step 2: 因子截面标准化 ----------------
        # 使用 z-score 标准化
        self.factor[self.fac_name] = self.factor.groupby('TradingDate')[self.fac_name]\
                                                .apply(lambda x: (x - x.mean()) / x.std())\
                                                .reset_index(level=0, drop=True)

        # ---------------- Step 3: 市值中性化处理 ----------------
        self.neutralize = neutralize
        if neutralize == 'cap' and market_value is not None:
            market_value = market_value.copy()
            df = pd.merge(self.factor, market_value, on=['TradingDate', 'Stkcd'], how='inner')

            def neutralize_by_cap(group):
                """
                对每个交易日的股票因子进行市值中性化处理
                """
                group['const'] = 1
                group['log_mv'] = np.log(group['MarketValue'])  # 市值取对数
                X = group[['log_mv', 'const']]
                y = group[self.fac_name]
                model = sm.OLS(y, X).fit()
                group['neutralized_factor'] = model.resid  # 残差作为中性化因子
                return group[['TradingDate', 'Stkcd', 'neutralized_factor']]

            # 按交易日分组应用中性化
            df_neutralized = df.groupby('TradingDate').apply(neutralize_by_cap)

            # 将中性化后的因子赋值回 self.factor
            self.factor = df_neutralized.rename(columns={'neutralized_factor': self.fac_name})

    @calculate_runtime
    def group_test(self, freq='monthly', save=False, period='all'):
        """执行分组回测，输出各分位净值曲线。

        Args:
            freq (str, optional): 调仓频率，支持 'daily'/'weekly'/'monthly'/'yearly' 等写法；
                'weekly' 实际使用周五（'W-Fri'），'monthly' 使用自然月（'MS'），默认值 'monthly'。
            save (bool, optional): 是否保存净值/收益率 CSV 与净值图 PNG，默认值 False。
            period (str, optional): 文件命名用的回测区间标识，默认值 'all'。

        Returns:
            pd.DataFrame: 行为分组标签，列为日期的净值表（转置后返回）。
        """

        # 统一频率
        freq = normalize_frequency(freq)

        # ---------------- Step 0: 根据频率对数据进行重采样 ----------------
        if freq == 'daily':
            df = self.factor.copy()
            df = df.merge(self.suspension_limit, on=['TradingDate', 'Stkcd'], how='right')
            df = df.merge(self.price, on=['TradingDate', 'Stkcd'], how='right')

        elif freq == 'weekly':  # 每周五调仓
            price = self.price.groupby([pd.Grouper(key='TradingDate', freq='W-Fri'), 'Stkcd'])['price'].last().reset_index()
            factor = self.factor.groupby([pd.Grouper(key='TradingDate', freq='W-Fri'), 'Stkcd'])[self.fac_name].last().reset_index()
            suspension_limit = self.suspension_limit.groupby([pd.Grouper(key='TradingDate', freq='W-Fri'), 'Stkcd'])[['Suspension', 'LimitStatus']].last().reset_index()
            df = factor.merge(suspension_limit, on=['TradingDate', 'Stkcd'], how='right')
            df = df.merge(price, on=['TradingDate', 'Stkcd'], how='right')

        elif freq == 'monthly':  # 自然月调仓
            price = self.price.groupby([pd.Grouper(key='TradingDate', freq='MS'), 'Stkcd'])['price'].last().reset_index()
            factor = self.factor.groupby([pd.Grouper(key='TradingDate', freq='MS'), 'Stkcd'])[self.fac_name].last().reset_index()
            suspension_limit = self.suspension_limit.groupby([pd.Grouper(key='TradingDate', freq='MS'), 'Stkcd'])[['Suspension', 'LimitStatus']].last().reset_index()
            df = factor.merge(suspension_limit, on=['TradingDate', 'Stkcd'], how='right')
            df = df.merge(price, on=['TradingDate', 'Stkcd'], how='right')

        # ---------------- Step 1: 计算每支股票收益率 ----------------
        df['ret'] = df.groupby('Stkcd')['price'].pct_change()        # 当期收益
        df['ret'] = df.groupby('Stkcd')['ret'].shift(-1)             # 下一期收益
        df['ret'] = df.groupby('Stkcd')['ret'].fillna(0)             # 缺失值填0
        market_index = df.groupby('TradingDate')['ret'].mean().reset_index()

        # ---------------- Step 2: 初始化分组标签与净值表 ----------------
        labels = [f'第{i}分位' for i in range(1, 11)]
        index_cols = labels + [self.market_index_name, 'Excess_+', 'Excess_-']
        net_value = pd.DataFrame(index=index_cols)
        group_ret = pd.DataFrame(index=index_cols)
        last_date = df['TradingDate'].max()
        net_value['first'] = 1 - 0.0001 - 0.00001
        previousday_df = None

        # ---------------- Step 3: 分组回测循环 ----------------
        for index, today_df in tqdm(df.groupby('TradingDate')):
            if index >= last_date:
                break
            
            # 因子值排名
            try:
                # --- 分位切分 ---
                today_df['quantile'] = today_df[self.fac_name].rank(method='first', pct=True)
                today_df['quantile_group'] = pd.qcut(today_df['quantile'], q=10, labels=labels, duplicates='drop')

                # --- 检查是否真的分成10个分位 ---
                unique_groups = today_df['quantile_group'].nunique(dropna=True)
                if unique_groups < 10:
                    raise ValueError(f"only {unique_groups} quantiles formed (less than 10)")

            except Exception as e:
                tqdm.write(f"Error at {index}: {e}")
                continue

            # 处理停牌和涨跌停股票
            if previousday_df is not None:
                today_df = today_df.merge(previousday_df, how='outer')
                mask = (today_df['Suspension'] == 1) | (today_df['LimitStatus'].isin([1, -1]))
                today_df.loc[mask, 'quantile_group'] = today_df.loc[mask, 'previous_group']
            else:
                today_df = today_df[(today_df['Suspension'] == 0) & (today_df['LimitStatus'] == 0)]

            # 计算每组平均收益
            r_mean = pd.DataFrame(index=labels).join(today_df.groupby('quantile_group')['ret'].mean())
            
            # 交易成本调整
            if previousday_df is not None:
                for q in labels:
                    group_q = today_df[today_df['quantile_group'] == q]
                    group_q['change'] = (group_q['quantile_group'] != group_q['previous_group'])
                    transaction_cost = group_q['change'].sum() / len(group_q) * (0.001 + 2*0.0001 + 2*0.00001)
                    r_mean.loc[q] -= transaction_cost

            # 市场收益赋值
            try:
                r_mean.loc[self.market_index_name] = market_index[market_index['TradingDate'] == index]['ret'].iloc[0]
            except:
                r_mean.loc[self.market_index_name] = 0

            # 计算超额收益
            r_mean.loc['Excess_+'] = r_mean.loc['第10分位'] - r_mean.loc[self.market_index_name]
            r_mean.loc['Excess_-'] = r_mean.loc['第1分位'] - r_mean.loc[self.market_index_name]

            r_mean = r_mean.rename(columns={'ret': index})
            group_ret = group_ret.join(r_mean, how='left')

            # 计算净值
            net_value = net_value.join((1 + r_mean).apply(lambda x: x * net_value.iloc[:, -1]), how='left')

            # 保存当日分组
            previousday_df = today_df[['Stkcd', 'quantile_group']].rename(columns={'quantile_group': 'previous_group'})

        # ---------------- Step 4: 调整列名和计算最终净值 ----------------
        column_name = net_value.columns.to_list()
        del column_name[0]
        column_name.append(last_date)
        net_value.columns = column_name

        net_value = net_value.T
        if net_value['第10分位'].mean() > net_value['第1分位'].mean():
            net_value['Excess'] = net_value['Excess_+']
        else:
            net_value['Excess'] = net_value['Excess_-']
        net_value = net_value.drop(columns=['Excess_+', 'Excess_-'])

        # ---------------- Step 5: 绘图 ----------------
        def set_plot_config():
            plt.rcParams["font.sans-serif"] = ["SimHei"]
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams['axes.unicode_minus'] = False
            global times_new_roman_path
            times_new_roman_path = fm.findfont(FontProperties(family='Times New Roman'))
        
        set_plot_config()
        
        colors = plt.cm.tab20.colors[:12]
        ax = net_value.plot(title=f'{self.fac_name}_{freq}_分层回测净值曲线', color=colors)
        if period == 'all':
            Bear_market = [
                ('2015-06-12', '2016-01-27'),
                ('2018-01-29', '2019-01-04'),
                ('2021-12-13', '2024-09-20')
            ]
            for bear_period in Bear_market:
                ax.axvspan(pd.to_datetime(bear_period[0]), pd.to_datetime(bear_period[1]), color='gray', alpha=0.3)
        ax.legend(loc='upper left')

        # ---------------- Step 6: 保存结果 ----------------
        if save:
            # 仅当需要保存时才创建目录
            os.makedirs(self.dir, exist_ok=True)
            if self.neutralize is None:
                net_value.T.to_csv(f"{self.dir}/{period}_无中性化_{freq}_{self.market_index_name}主板_分组净值.csv")
                group_ret.to_csv(f"{self.dir}/{period}_无中性化_{freq}_{self.market_index_name}主板_分组收益率.csv")
                plt.savefig(f"{self.dir}/{period}_无中性化_{freq}_{self.market_index_name}主板_分组净值.png", dpi=250, bbox_inches='tight')
            elif self.neutralize == 'cap':
                net_value.T.to_csv(f"{self.dir}/{period}_市值中性化_{freq}_{self.market_index_name}主板_分组净值.csv")
                group_ret.to_csv(f"{self.dir}/{period}_市值中性化_{freq}_{self.market_index_name}主板_分组收益率.csv")
                plt.savefig(f"{self.dir}/{period}_市值中性化_{freq}_{self.market_index_name}主板_分组净值.png", dpi=250, bbox_inches='tight')

        self.net_value = net_value.T
        return net_value.T


    def aly_group_test(self, freq='monthly', save=False, risk_free_rate=0.02, period='all'):
        """分析分组回测结果，输出关键绩效指标。

        Args:
            freq (str, optional): 调仓频率：'daily'/'weekly'/'monthly'/'yearly'，影响年化倍数，默认值 'monthly'。
            save (bool, optional): 是否保存分析结果 CSV，默认值 False。
            risk_free_rate (float, optional): 年化无风险利率，默认值 0.02。
            period (str, optional): 文件命名用的回测区间标识，默认值 'all'。

        Returns:
            None: 打印表格到控制台，并可选保存 CSV。
        """

        # ---------------- Step 0: 获取净值数据并计算收益率 ----------------
        # 统一频率
        freq = normalize_frequency(freq)
        df = self.net_value.T
        pct = df.pct_change().dropna(how='any')  # 每期收益率

        # ---------------- Step 1: 单调性分析 ----------------
        test = pd.DataFrame({
            '分层累计收益率': df.T.iloc[:10, -1],
            'Group': list(range(1, 11))
        })
        monotonicity = test.corr().iloc[0, 1]
        print("单调性分析：", monotonicity)

        # ---------------- Step 2: 根据频率设置年化倍数 ----------------
        times = 52  # 默认周频
        if freq == 'monthly':
            times = 12
        elif freq == 'daily':
            times = 252

        # ---------------- Step 3: 年化指标计算 ----------------
        groupStd = pct.std() * np.sqrt(times)  # 年化标准差
        temp = df.iloc[-1, :] / df.iloc[0, :]
        groupYield = (np.power(np.abs(temp), times / len(pct)) - 1) * np.sign(temp)  # 几何平均年化收益率
        groupSharp = (groupYield - risk_free_rate) / groupStd  # 夏普比率

        # ---------------- Step 4: 最大回撤计算 ----------------
        def MaxDrawdown(return_list):
            return ((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list)).max()

        groupMaxDrawdown = df.apply(MaxDrawdown, axis=0)

        # ---------------- Step 5: 累计收益率计算 ----------------
        groupCumulativeReturn = df.iloc[-1, :] / df.iloc[0, :] - 1

        # ---------------- Step 6: 汇总结果 ----------------
        result = pd.concat([groupCumulativeReturn, groupYield, groupStd, groupSharp, groupMaxDrawdown], axis=1)
        result.columns = ['CumulativeReturn', 'anlYield', 'anlStd', 'Sharp', 'MaxDrawdown']

        pretty_print_df(result)

        # ---------------- Step 7: 保存结果 ----------------
        if save:
            # 仅当需要保存时才创建目录
            os.makedirs(self.dir, exist_ok=True)
            result['单调性'] = monotonicity
            if self.neutralize is None:
                result.to_csv(f"{self.dir}/{period}_无中性化_{freq}_{self.market_index_name}主板_分组回测结果.csv")
            elif self.neutralize == 'cap':
                result.to_csv(f"{self.dir}/{period}_市值中性化_{freq}_{self.market_index_name}主板_分组回测结果.csv")


def run_bt(
    factor_path: Path | str | list[Path | str],
    result_dir: Path | str,
    adj_close_path: Path | str,
    suspension_limit_path: Path | str,
    stock_pool_path: Path | str,
    market_value_path: Path | str,
    neutralize: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    frequency: str = "monthly",
    save_backtest_result: bool = False,
):
    """对一个或一组因子执行完整回测流程。

    流程包括：因子处理、分组回测以及回测结果分析。

    Args:
        factor_path (Path | str | list[Path | str]): 因子文件或目录，或其列表。
        result_dir (Path | str): 结果保存根目录。
        adj_close_path (Path | str): 复权收盘价 parquet 路径。
        suspension_limit_path (Path | str): 停牌/涨跌停 parquet 路径。
        stock_pool_path (Path | str): 股票池 parquet 路径（矩阵形式）。
        market_value_path (Path | str): 市值 parquet 路径，需含 ['TradingDate','Stkcd','MarketValue']。
        neutralize (str | None, optional): 中性化方式，当前支持 'cap' 或 None，默认值 None。
        start_date (str | None, optional): 起始日期，支持 'YYYY'/'YYYY-MM'/'YYYY-MM-DD'，如果为None则使用因子最早日期，默认值 None。
        end_date (str | None, optional): 结束日期，支持 'YYYY'/'YYYY-MM'/'YYYY-MM-DD'，如果为None则使用因子最晚日期，默认值 None。
        frequency (str, optional): 回测频率，'daily'/'weekly'/'monthly'/'yearly'，默认值 'monthly'。
        save_backtest_result (bool, optional): 是否保存净值/收益率与分析结果，默认值 False。

    Raises:
        ValueError: 传入路径无效或类型非法时。
        FileNotFoundError: 未找到任何有效因子文件时。
        TypeError: ``start_date`` 类型不符合要求时。
    """

    # 标准化起始日期
    def _normalize_start_date(date_str: str) -> str:
        parts = date_str.split("-")
        if len(parts) == 1:  # 年
            return f"{parts[0]}-01-01"
        elif len(parts) == 2:  # 年月
            return f"{parts[0]}-{parts[1]}-01"
        else:  # 年月日
            dt = parser.parse(date_str)
            return dt.strftime("%Y-%m-%d")

    # ---------------- Step 0: 读取固定数据 ----------------
    close_price = pd.read_parquet(adj_close_path, engine="pyarrow")
    suspension_limit = pd.read_parquet(suspension_limit_path, engine="pyarrow")
    stock_pool = pd.read_parquet(stock_pool_path, engine="pyarrow")
    market_value = pd.read_parquet(market_value_path, engine="pyarrow")[["TradingDate", "Stkcd", "MarketValue"]]
    market_value["TradingDate"] = pd.to_datetime(market_value["TradingDate"])

    # ---------------- Step 1: 获取因子文件列表 ----------------
    factor_path_list = factor_path if isinstance(factor_path, list) else [factor_path]
    factor_files = []

    for path in factor_path_list:
        path = Path(path)
        if path.is_dir():
            for f in path.iterdir():
                if f.suffix in (".csv", ".parquet", ".pq"):
                    factor_files.append(f)
        elif path.is_file():
            factor_files.append(path)
        else:
            raise ValueError(f"传入路径 {path} 既不是文件也不是目录")

    if not factor_files:
        raise FileNotFoundError("未找到任何有效因子文件")

    print("Factor files to backtest:")
    for idx, f in enumerate(factor_files):
        print(f"[{idx + 1}] {f}")
    
    # ---------------- Step 2: 遍历因子文件 ----------------
    for idx, fpath in enumerate(factor_files):
        fname = fpath.name
        if fname.endswith(".csv"):
            factor_name = fname[:-4]
            df = pd.read_csv(fpath)
        elif fname.endswith(".parquet"):
            factor_name = fname[:-8]
            df = pd.read_parquet(fpath, engine="pyarrow")
        elif fname.endswith(".pq"):
            factor_name = fname[:-3]
            df = pd.read_parquet(fpath, engine="pyarrow")
        else:
            continue

        print(f"\n{'=' * 20} backtesting {factor_name} ({idx + 1}/{len(factor_files)}) {'=' * 20}")

        result_dir_f = Path(result_dir) / factor_name
        # 仅当需要保存时才创建目录
        if save_backtest_result:
            result_dir_f.mkdir(parents=True, exist_ok=True)

        # ---------------- Step 3: 处理 start_date 和 end_date ----------------
        if start_date is not None and not isinstance(start_date, str):
            raise TypeError("start_date 必须是 str 或 None")
        if end_date is not None and not isinstance(end_date, str):
            raise TypeError("end_date 必须是 str 或 None")

        # 获取因子的最早和最晚日期
        factor_start_date = df["TradingDate"].min().strftime("%Y-%m-%d")
        factor_end_date = df["TradingDate"].max().strftime("%Y-%m-%d")

        # 处理 start_date
        if start_date is None:
            effective_start_date = factor_start_date
        else:
            effective_start_date = _normalize_start_date(start_date)
            if pd.to_datetime(effective_start_date) < pd.to_datetime(factor_start_date):
                effective_start_date = factor_start_date

        # 处理 end_date
        if end_date is None:
            effective_end_date = factor_end_date
        else:
            effective_end_date = _normalize_start_date(end_date)
            if pd.to_datetime(effective_end_date) > pd.to_datetime(factor_end_date):
                effective_end_date = factor_end_date

        period_label = f"from_{effective_start_date}_to_{effective_end_date}"

        test = back_test(
            df, close_price, factor_name, result_dir_f, "all", suspension_limit, start_date=effective_start_date, end_date=effective_end_date
        )

        test.factor_process(stock_pool=stock_pool, market_value=market_value, neutralize=neutralize)

        print(str(test))
        print(format_single_line_info(f"Frequency: {frequency}"))
        
        test.group_test(freq=frequency, save=save_backtest_result, period=period_label)
        test.aly_group_test(freq=frequency, save=save_backtest_result, period=period_label)

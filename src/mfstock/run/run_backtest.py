# 标准库
import warnings
from pathlib import Path
import argparse

warnings.simplefilter("ignore")

# 本地模块
from mfstock.backtest import run_bt
from mfstock.utils.format_display import format_box_title
from mfstock.utils.misc import load_config, parse_project_path


def get_factor_files(factor_path: Path | str | list[Path | str]) -> list[Path]:
    """
    根据 factor_path 获取因子文件列表，支持单文件、目录或列表

    Args:
        factor_path (Path | str | list[Path | str]): 因子文件或目录路径

    Returns:
        list[Path]: 有效因子文件列表
    """
    valid_ext = {".csv", ".parquet", ".pq"}
    paths = factor_path if isinstance(factor_path, list) else [factor_path]
    factor_files: list[Path] = []

    for p in paths:
        p = Path(p)
        if p.is_file() and p.suffix in valid_ext:
            factor_files.append(p)
        elif p.is_dir():
            files = [f for f in p.glob("*") if f.suffix in valid_ext]
            if not files:
                raise FileNotFoundError(f"目录中未找到有效因子文件: {p}")
            factor_files.extend(files)
        else:
            raise FileNotFoundError(f"路径无效或非支持文件类型: {p}")

    return factor_files


def main():
    # ---------- 命令行参数 ----------
    parser = argparse.ArgumentParser(description="Backtest factors")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/backtest_config.yaml",
    )
    args = parser.parse_args()
    config_path = Path(args.config)

    # ---------- 加载配置 ----------
    if not config_path.exists():
        raise FileNotFoundError(
            f"配置文件 {config_path} 不存在，请先从模板复制或创建"
        )
    config = load_config(config_path)

    # ---------- 获取路径 ----------
    paths_cfg = config.get("paths_config", {})
    required_paths = [
        "adj_close_path",
        "suspension_limit_path",
        "stock_pool_path",
        "market_value_path",
        "factor_path",
        "result_dir"
    ]

    for p in required_paths:
        if p not in paths_cfg:
            raise ValueError(f"{config_path.name} 的 paths_config 未指定 {p}")

    adj_close_path = parse_project_path(paths_cfg["adj_close_path"])
    suspension_limit_path = parse_project_path(paths_cfg["suspension_limit_path"])
    stock_pool_path = parse_project_path(paths_cfg["stock_pool_path"])
    market_value_path = parse_project_path(paths_cfg["market_value_path"])

    # 支持 factor_path 为单个路径或路径列表
    raw_factor_path = paths_cfg["factor_path"]
    if isinstance(raw_factor_path, list):
        factor_path = [parse_project_path(p) for p in raw_factor_path]
    else:
        factor_path = parse_project_path(raw_factor_path)

    result_dir = parse_project_path(paths_cfg["result_dir"])

    # ---------- 获取回测方法参数 ----------
    method_cfg = config.get("backtest_config", {})
    neutralize = method_cfg.get("neutralize", None)
    start_date = method_cfg.get("start_date", "2020")
    end_date = method_cfg.get("end_date", None)
    backtest_frequency = method_cfg.get("backtest_frequency") or "monthly"
    save_backtest_result = method_cfg.get("save_backtest_result", False)

    # ---------- 获取因子文件 ----------
    factor_files = get_factor_files(factor_path)

    # ---------- 回测 ----------
    print(format_box_title("Backtesting factors"))
    run_bt(
        factor_path=factor_files,
        result_dir=result_dir,
        adj_close_path=adj_close_path,
        suspension_limit_path=suspension_limit_path,
        stock_pool_path=stock_pool_path,
        market_value_path=market_value_path,
        neutralize=neutralize,
        start_date=start_date,
        end_date=end_date,
        frequency=backtest_frequency,
        save_backtest_result=save_backtest_result,
    )


if __name__ == "__main__":
    main()

import time
import yaml
from pathlib import Path

__all__ = [
    'parse_project_path',
    'get_project_root',
    'load_config',
    'get_datefmt_from_freq',
    'normalize_frequency',
    'calculate_runtime',
]

class ProjectPathParser:
    _instance = None
    _project_root = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if self._project_root is None:
            self._load_project_root()

    def _load_project_root(self):
        # 尝试从当前工作目录加载
        config_path = Path("configs/project_config.yaml")
        
        if not config_path.exists():
            current_file = Path(__file__).resolve()
            potential_root = current_file.parents[2]
            config_path = potential_root / "configs/project_config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError("Could not find configs/project_config.yaml")
            
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            self._project_root = Path(config["project_root"])

    def parse(self, path_cfg: dict | str) -> Path:
        if isinstance(path_cfg, str):
            return self._project_root / path_cfg
        elif isinstance(path_cfg, dict):
            fmt = path_cfg.get("fmt")
            if not fmt:
                raise ValueError(f"Path config dict must contain 'fmt' key: {path_cfg}")
            
            kwargs = {k: v for k, v in path_cfg.items() if k != "fmt"}
            
            try:
                relative_path = fmt.format(**kwargs)
            except KeyError as e:
                 raise KeyError(f"Missing key for path formatting: {e.args[0]} in {fmt}")
                 
            return self._project_root / relative_path
        else:
            raise TypeError(f"Unexpected type for path config: {type(path_cfg)}")

    def get_root(self) -> Path:
        return self._project_root

def parse_project_path(path_cfg: dict | str) -> Path:
    """
    解析项目配置路径的统一接口。
    自动加载项目根目录配置，并解析相对路径。
    """
    return ProjectPathParser.get_instance().parse(path_cfg)


def get_project_root() -> Path:
    """
    获取项目根目录。
    """
    return ProjectPathParser.get_instance().get_root()


def load_config(path: str | Path) -> dict:
    """
    加载 YAML 配置文件

    Args:
        path (str | Path): YAML 配置文件路径

    Returns:
        dict: 配置内容
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_datefmt_from_freq(frequency: str) -> str:
    """根据频率返回日期格式化字符串。

    Args:
        frequency (str): 频率类型：'yearly'、'monthly'、'weekly'、'daily' 

    Returns:
        str: 对应的 ``strftime`` 格式字符串。
    """
    frequency = normalize_frequency(frequency)
    if frequency == "yearly":
        return "%Y"
    elif frequency == "monthly":
        return "%Y%m"
    elif frequency == "weekly":
        return "%Y%m%d"  # 可根据需要自定义周格式
    else:  # daily
        return "%Y%m%d"
    
def normalize_frequency(freq: str) -> str:
    """将多种写法的频率值规范化为完整小写形式。

    支持的别名（大小写不敏感）：
    - 日频：'d', 'day', 'daily' -> 'daily'
    - 周频：'w', 'week', 'weekly' -> 'weekly'
    - 月频：'m', 'mon', 'month', 'monthly' -> 'monthly'
    - 年频：'y', 'year', 'yearly' -> 'yearly'

    其他输入将引发 ValueError 异常。    python -c "from quant_library.utils.miscs import build_factor_name, parse_factor_name, rank_label; fn=build_factor_name('XGB', 'optuna', 'window(10y+1m)', 'AE60', rank_label(rank_label(True, True), False))"
    """
    if not isinstance(freq, str):
        raise ValueError(f"频率必须是字符串类型，而不是 {type(freq)}")
    
    f = freq.strip().lower()
    
    # 映射字典 - 返回完整小写形式
    mapping = {
        'd': 'daily', 'day': 'daily', 'daily': 'daily',
        'w': 'weekly', 'week': 'weekly', 'weekly': 'weekly',
        'm': 'monthly', 'mon': 'monthly', 'month': 'monthly', 'monthly': 'monthly',
        'y': 'yearly', 'year': 'yearly', 'yearly': 'yearly'
    }
    
    result = mapping.get(f)
    if result is None:
        raise ValueError(f"不支持的频率值：'{freq}'。支持的值为：daily(d), weekly(w), monthly(m), yearly(y)")
    
    return result

def calculate_runtime(func):
    """装饰器：计算函数运行时间并打印。

    Args:
        func (Callable): 被装饰的函数。

    Returns:
        Callable: 包装后的函数，执行后会打印耗时信息。
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        print(f">> {func.__name__} running time: {runtime:.2f} seconds\n")
        return result

    return wrapper

"""控制台输出格式化工具。

提供用于格式化单行与盒式标题、美化打印 pandas DataFrame、以及渲染多行信息块的便捷函数。
文档字符串统一采用 Google 风格。
"""
import pandas as pd


__all__ = [
    "format_single_line_title",
    "format_single_line_info",
    "format_box_title",
    "pretty_print_df",
    "format_multiline_block"
]


def format_single_line_title(text: str, symbol: str = '=') -> str:
    """格式化单行标题。

    在标题左右两端填充固定数量的符号。

    Args:
        text (str): 标题内容。
        symbol (str, optional): 填充符号，默认值 '='。

    Returns:
        str: 格式化后的标题字符串。
    """
    return f"{symbol * 20} {text} {symbol * 20}"


def format_single_line_info(text: str, symbol: str = '>') -> str:
    """格式化单行信息。

    在信息左侧填充固定数量的符号。

    Args:
        text (str): 信息内容。
        symbol (str, optional): 填充符号，默认值 '-'。

    Returns:
        str: 格式化后的信息字符串。
    """
    return f"{symbol * 2} {text}"


def format_box_title(
    title: str,
    horizontal: str = '-',
    vertical: str = '|',
    corner: str = '+',
    padding: int = 1
) -> str:
    """生成使用自定义符号的盒式标题。

    Args:
        title (str): 显示的标题文本。
        horizontal (str, optional): 横线符号，默认值 '-'。
        vertical (str, optional): 竖线符号，默认值 '|'。
        corner (str, optional): 角符号，默认值 '+'。
        padding (int, optional): 标题左右的空格数量，默认值 1。

    Returns:
        str: 由多行组成的盒式标题字符串。
    """
    # 计算横线长度：标题长度 + 左右 padding + 两侧空格
    line_len = len(title) + padding * 2
    top_bottom = corner + horizontal * line_len + corner
    middle = f"{vertical}{' ' * padding}{title}{' ' * padding}{vertical}"
    return f"{top_bottom}\n{middle}\n{top_bottom}"


def pretty_print_df(
    df: pd.DataFrame,
    float_format: str = "%.6f",
    max_rows: int | None = None,
    max_cols: int | None = None,
    width: int = 120,
):
    """以更友好的方式打印 DataFrame。

    若安装了 tabulate，则优先使用其进行美化输出；否则退化为 pandas 自带的字符串输出。

    Args:
        df (pd.DataFrame): 要打印的数据框。
        float_format (str, optional): 浮点数格式化字符串，如 "%.4f"，默认值 "%.6f"。
        max_rows (int | None, optional): 最大显示行数，默认值 None（不限制）。
        max_cols (int | None, optional): 最大显示列数，默认值 None（不限制）。
        width (int, optional): 输出总宽度，用于控制换行，默认值 120。

    Returns:
        None: 该函数直接打印，不返回值。
    """
    # 格式化所有float列为字符串，保留6位小数
    df_fmt = df.copy()
    float_cols = df_fmt.select_dtypes(include=['float', 'float32', 'float64']).columns
    for col in float_cols:
        df_fmt[col] = df_fmt[col].apply(lambda x: (float_format % x) if pd.notnull(x) else "")
    try:
        from tabulate import tabulate
        print(tabulate(
            df_fmt,
            headers='keys',
            tablefmt='pretty',
            showindex=True
        ))
    except ImportError:
        with pd.option_context(
            'display.max_rows', max_rows or None,
            'display.max_columns', max_cols or None,
            'display.width', width
        ):
            print(df_fmt.to_string(index=True))


def format_multiline_block(
    title: str,
    lines: list[str],
    indent: int = 4,
    bullet: str = '-'
) -> str:
    """将多行信息格式化为带缩进与项目符号的文本块。

    Args:
        title (str): 标题行文本。
        lines (list[str]): 每行的内容列表。
        indent (int, optional): 每行前缩进空格数，默认值 4。
        bullet (str, optional): 每行前的项目符号，默认值 '-'。

    Returns:
        str: 格式化后的多行字符串。
    """
    indent_space = ' ' * indent
    formatted_lines = [f"{indent_space}{bullet} {line}" for line in lines]
    block = f"{format_single_line_info(title)}\n" + "\n".join(formatted_lines)
    return block

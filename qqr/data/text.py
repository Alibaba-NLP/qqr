import re

char_pattern = re.compile(r"[0-9a-zA-Z\u4e00-\u9fff]")
cjk_char_pattern = re.compile(r"[一-龥]")


def load_text(data_file, split: bool = False) -> str | list[str]:
    with open(data_file) as f:
        data = f.read()

    if split:
        data = data.splitlines()

    return data


def to_bool(x) -> bool:
    if isinstance(x, str):
        return x.lower() in ["1", "true", "yes"]
    else:
        return bool(x)


def check_str(v) -> bool:
    return isinstance(v, str) and v.strip()


def get_chars(text: str) -> str:
    return "".join(char_pattern.findall(text))


def get_cjk_chars(text: str) -> str:
    return "".join(cjk_char_pattern.findall(text))


def is_cjk(text: str) -> bool:
    return bool(cjk_char_pattern.search(text))


def truncate_text(text: str, max_len: int = 5000) -> str:
    if len(text) <= max_len:
        return text

    head_len = max_len // 2
    tail_len = max_len // 2

    head_part = text[:head_len]
    head_matches = list(re.finditer(r"\s", head_part))
    if head_matches:
        head_end_index = head_matches[-1].start()
    else:
        head_end_index = head_len
    head = text[:head_end_index]

    tail_part = text[-tail_len:]
    tail_match = re.search(r"\s", tail_part)
    if tail_match:
        tail_start_index_in_part = tail_match.start()
        tail_start_index = len(text) - tail_len + tail_start_index_in_part
        tail = text[tail_start_index:].lstrip()
    else:
        tail = tail_part

    truncated_chars = len(text) - len(head) - len(tail)
    ellipsis = f"\n\n... [内容已截断，共省略 {truncated_chars} 字符] ...\n\n"

    return head + ellipsis + tail

import re
from typing import Optional, List


def get_content_from_tag(
    content: str,
    tag: str,
    default_value: Optional[str] = None
) -> Optional[str]:
    """
    extract the content from the first specified tag
    """
    if not content:
        return default_value

    pattern = rf"<{tag}>(.*?)(?=(</{tag}>|<\w+|$))"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return default_value

def get_all_content_from_tag(
    content: str,
    tag: str
) -> List[str]:
    """
    extract all the content from the specified tag
    """
    if not content:
        return []

    pattern = rf"<{tag}>(.*?)(?=(</{tag}>|<\w+|$))"
    matches = re.findall(pattern, content, re.DOTALL)
    if matches:
        matches = [match[0].strip() for match in matches]
        matches = [match for match in matches if match]
        return matches
    return []
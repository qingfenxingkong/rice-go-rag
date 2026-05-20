"""
go_ner/obo_parser.py
解析 go.obo 文件，构建 GO 术语字典。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class GOEntry:
    go_id: str
    name: str
    namespace: str
    definition: str = ""
    synonyms: List[str] = field(default_factory=list)
    is_obsolete: bool = False
    alt_ids: List[str] = field(default_factory=list)

    def all_names(self) -> List[str]:
        """返回该术语的所有名称（标准名 + 同义词）。"""
        return [self.name] + self.synonyms


def parse_obo(obo_path: str | Path) -> Dict[str, GOEntry]:
    """
    解析 OBO 文件，返回 {go_id: GOEntry} 字典。
    跳过已废弃（is_obsolete: true）的术语。
    """
    obo_path = Path(obo_path)
    entries: Dict[str, GOEntry] = {}

    current: Optional[Dict] = None
    in_term = False

    with obo_path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip()

            if line == "[Term]":
                # 保存上一个 Term
                if current and not current.get("is_obsolete"):
                    _save_entry(current, entries)
                current = {}
                in_term = True
                continue

            if line.startswith("[") and line != "[Term]":
                # Typedef 等其他块
                if current and not current.get("is_obsolete"):
                    _save_entry(current, entries)
                current = None
                in_term = False
                continue

            if not in_term or current is None:
                continue

            if ": " not in line:
                continue

            key, _, value = line.partition(": ")
            key = key.strip()
            value = value.strip()

            if key == "id":
                current["go_id"] = value
            elif key == "name":
                current["name"] = value
            elif key == "namespace":
                current["namespace"] = value
            elif key == "def":
                # def: "..." [refs]
                m = re.match(r'"(.+?)"', value)
                current["definition"] = m.group(1) if m else value
            elif key == "synonym":
                m = re.match(r'"(.+?)"', value)
                if m:
                    current.setdefault("synonyms", []).append(m.group(1))
            elif key == "alt_id":
                current.setdefault("alt_ids", []).append(value)
            elif key == "is_obsolete" and value == "true":
                current["is_obsolete"] = True

    # 最后一个 Term
    if current and not current.get("is_obsolete"):
        _save_entry(current, entries)

    print(f"[OBO Parser] 共解析 {len(entries)} 个有效 GO 术语")
    return entries


def _save_entry(current: dict, entries: dict) -> None:
    go_id = current.get("go_id", "")
    name = current.get("name", "")
    if not go_id or not name:
        return
    entry = GOEntry(
        go_id=go_id,
        name=name,
        namespace=current.get("namespace", ""),
        definition=current.get("definition", ""),
        synonyms=current.get("synonyms", []),
        is_obsolete=current.get("is_obsolete", False),
        alt_ids=current.get("alt_ids", []),
    )
    entries[go_id] = entry
    # alt_id 也指向同一个 Entry
    for alt in entry.alt_ids:
        entries[alt] = entry

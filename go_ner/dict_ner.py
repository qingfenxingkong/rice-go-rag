"""
go_ner/dict_ner.py
方案一：基于字典的 GO 术语实体识别与概念标准化

策略：
  1. 精确匹配（大小写不敏感）
  2. 同义词匹配
  3. 模糊匹配（基于 difflib，可配置阈值）
  4. GO ID 直接识别（如文本中出现 GO:0006950）
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from .obo_parser import GOEntry, parse_obo


@dataclass
class NERResult:
    """单个识别结果。"""
    span: str           # 原文中识别到的文本片段
    start: int          # 在原文中的起始位置
    end: int            # 在原文中的结束位置
    go_id: str          # 标准化后的 GO ID
    go_name: str        # 标准名称
    namespace: str      # 命名空间
    match_type: str     # 匹配类型: exact / synonym / fuzzy / go_id
    score: float        # 匹配置信度 (1.0 = 精确)


class DictNER:
    """
    基于字典的 GO 术语 NER + 概念标准化。

    Args:
        obo_path:         go.obo 文件路径
        fuzzy_threshold:  模糊匹配阈值 (0~1)，默认 0.85
        use_fuzzy:        是否启用模糊匹配（较慢）
        entries:          可选，直接传入 {go_id: GOEntry} 字典（用于压缩模式）
    """

    # 匹配 GO ID 的正则
    _GO_ID_RE = re.compile(r'\bGO:\d{7}\b', re.IGNORECASE)

    def __init__(
        self,
        obo_path: str,
        fuzzy_threshold: float = 0.85,
        use_fuzzy: bool = True,
        entries: Optional[Dict[str, 'GOEntry']] = None,
    ) -> None:
        self.fuzzy_threshold = fuzzy_threshold
        self.use_fuzzy = use_fuzzy

        if entries is not None:
            print(f"[DictNER] 使用外部传入的字典（{len(entries)} 个术语）")
            self._entries = entries
        else:
            print("[DictNER] 加载 OBO 文件...")
            self._entries = parse_obo(obo_path)

        # 构建名称 -> go_id 的倒排索引（小写）
        print("[DictNER] 构建名称索引...")
        self._name_index: Dict[str, str] = {}
        self._all_names: List[Tuple[str, str]] = []

        for go_id, entry in self._entries.items():
            if go_id != entry.go_id:
                continue
            for n in entry.all_names():
                lower = n.lower()
                if lower not in self._name_index:
                    self._name_index[lower] = go_id
                    self._all_names.append((lower, go_id))

        print(f"[DictNER] 索引构建完成，共 {len(self._name_index)} 个名称")

    def recognize(self, text: str) -> List[NERResult]:
        """
        在文本中识别 GO 术语，返回所有识别结果（按位置排序）。
        改进：去掉停用词边界污染，清理标点。
        """
        # 停用词集合
        stopwords = {'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}

        results: List[NERResult] = []

        # 步骤1：识别文本中直接出现的 GO ID
        for m in self._GO_ID_RE.finditer(text):
            go_id = m.group(0).upper()
            go_id = "GO:" + go_id[3:]
            entry = self._entries.get(go_id)
            if entry:
                results.append(NERResult(
                    span=m.group(0),
                    start=m.start(),
                    end=m.end(),
                    go_id=entry.go_id,
                    go_name=entry.name,
                    namespace=entry.namespace,
                    match_type="go_id",
                    score=1.0,
                ))

        # 步骤2：基于名称字典的滑动窗口匹配
        tokens = re.findall(r'[\w\-\/]+', text.lower())
        token_spans = list(re.finditer(r'[\w\-\/]+', text))

        max_ngram = 8
        matched_ranges = set()

        # 优先匹配更长的短语，再到短的
        for n in range(max_ngram, 0, -1):
            for i in range(len(tokens) - n + 1):
                phrase = " ".join(tokens[i:i+n])
                start = token_spans[i].start()
                end = token_spans[i+n-1].end()

                # 检查是否已被更长的匹配覆盖
                if any(s <= start < e or s < end <= e for s, e in matched_ranges):
                    continue

                # 去掉前后停用词
                phrase_clean = phrase.strip()
                words = phrase_clean.split()
                while words and words[0] in stopwords:
                    words.pop(0)
                while words and words[-1] in stopwords:
                    words.pop()
                if not words:
                    continue
                phrase_clean = " ".join(words)

                # 调整 span 范围（去掉停用词后）
                if phrase_clean != phrase:
                    # 重新定位 start/end
                    idx_start = phrase.find(phrase_clean)
                    idx_end = idx_start + len(phrase_clean)
                    # 从 token_spans 里找对应的字符位置
                    first_word_start = token_spans[i].start()
                    last_word_end = token_spans[i+n-1].end()
                    # 简单方案：在原文本里找 phrase_clean
                    m = re.search(re.escape(phrase_clean), text[first_word_start:last_word_end])
                    if m:
                        start = first_word_start + m.start()
                        end = first_word_start + m.end()
                    else:
                        continue

                # 精确匹配
                go_id = self._name_index.get(phrase_clean)
                if go_id:
                    entry = self._entries[go_id]
                    match_type = "exact" if phrase_clean == entry.name.lower() else "synonym"
                    results.append(NERResult(
                        span=text[start:end],
                        start=start, end=end,
                        go_id=entry.go_id,
                        go_name=entry.name,
                        namespace=entry.namespace,
                        match_type=match_type,
                        score=1.0,
                    ))
                    matched_ranges.add((start, end))
                    continue

                # 模糊匹配（仅对 2 词以上短语）
                if self.use_fuzzy and len(phrase_clean.split()) >= 2:
                    best_score = 0.0
                    best_goid = None
                    for name, gid in self._all_names:
                        if abs(len(name) - len(phrase_clean)) > 10:
                            continue
                        score = SequenceMatcher(None, phrase_clean, name).ratio()
                        if score > best_score:
                            best_score = score
                            best_goid = gid
                    if best_score >= self.fuzzy_threshold and best_goid:
                        entry = self._entries[best_goid]
                        results.append(NERResult(
                            span=text[start:end],
                            start=start, end=end,
                            go_id=entry.go_id,
                            go_name=entry.name,
                            namespace=entry.namespace,
                            match_type="fuzzy",
                            score=round(best_score, 3),
                        ))
                        matched_ranges.add((start, end))

        # 按位置排序，去重
        results.sort(key=lambda r: r.start)
        seen = set()
        unique = []
        for r in results:
            key = (r.start, r.end)
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique

    def normalize(self, term: str) -> Optional[GOEntry]:
        """
        将一个术语字符串标准化为 GOEntry。
        先精确匹配，再模糊匹配。
        """
        lower = term.lower().strip()

        # 精确匹配
        go_id = self._name_index.get(lower)
        if go_id:
            return self._entries[go_id]

        # GO ID 直接查找
        m = self._GO_ID_RE.match(term.strip())
        if m:
            go_id = "GO:" + m.group(0)[3:].upper()
            return self._entries.get(go_id)

        # 模糊匹配
        if self.use_fuzzy:
            best_score = 0.0
            best_entry = None
            for name, gid in self._all_names:
                score = SequenceMatcher(None, lower, name).ratio()
                if score > best_score:
                    best_score = score
                    best_entry = self._entries[gid]
            if best_score >= self.fuzzy_threshold:
                return best_entry

        return None

    def format_results(self, text: str, results: List[NERResult]) -> str:
        """将识别结果格式化为可读字符串。"""
        if not results:
            return "未识别到任何 GO 术语。"
        lines = [f"文本：{text[:100]}{'...' if len(text)>100 else ''}\n"]
        lines.append(f"共识别到 {len(results)} 个 GO 术语：\n")
        for r in results:
            lines.append(
                f"  [{r.match_type:8s}] \"{r.span}\"\n"
                f"             -> {r.go_id} | {r.go_name}\n"
                f"             命名空间: {r.namespace} | 置信度: {r.score:.3f}\n"
            )
        return "\n".join(lines)

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.util import ngrams

from .obo_parser import GOEntry, parse_obo


@dataclass
class NLTKNERResult:
    span: str
    start: int
    end: int
    go_id: str
    go_name: str
    namespace: str
    match_type: str
    score: float


class NLTKNER:
    """
    基于 NLTK 的轻量对比方案。

    思路：
      1) 用 NLTK 进行英文/混合文本分词
      2) 枚举 1-5 gram 候选短语
      3) 用 GO 名称/同义词字典做精确匹配

    这是一个传统 NLP baseline，目标是作为对比方案，而不是最优模型。
    """

    _GO_ID_RE = re.compile(r'\bGO:\d{7}\b', re.IGNORECASE)
    _TOKEN_RE = re.compile(r'[\w\-/]+')
    _STOPWORDS = {
        'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'and', 'or', 'that', 'this', 'which', 'it', 'its', 'into', 'under',
    }

    def __init__(self, obo_path: str, max_ngram: int = 5) -> None:
        print('[NLTKNER] 加载 OBO 文件...')
        self._entries: Dict[str, GOEntry] = parse_obo(obo_path)
        self.max_ngram = max_ngram

        print('[NLTKNER] 构建名称索引...')
        self._name_index: Dict[str, GOEntry] = {}
        for go_id, entry in self._entries.items():
            if go_id != entry.go_id:
                continue
            for name in entry.all_names():
                lower = name.lower().strip()
                if lower and lower not in self._name_index:
                    self._name_index[lower] = entry
        print(f'[NLTKNER] 索引构建完成，共 {len(self._name_index)} 个名称')

    def _token_spans(self, text: str) -> List[Tuple[str, int, int]]:
        spans: List[Tuple[str, int, int]] = []
        for m in self._TOKEN_RE.finditer(text):
            token = m.group(0)
            # 用 NLTK 的分词函数进行标准化切分；这里保留原 token 位置
            pieces = [p for p in wordpunct_tokenize(token) if p.strip()]
            if len(pieces) == 1:
                spans.append((token.lower(), m.start(), m.end()))
            else:
                # 若 NLTK 把 token 再切开，则退回原 token
                spans.append((token.lower(), m.start(), m.end()))
        return spans

    def recognize(self, text: str) -> List[NLTKNERResult]:
        results: List[NLTKNERResult] = []
        matched_ranges = set()

        for m in self._GO_ID_RE.finditer(text):
            go_id = m.group(0).upper()
            entry = self._entries.get(go_id)
            if entry:
                results.append(NLTKNERResult(
                    span=m.group(0), start=m.start(), end=m.end(),
                    go_id=entry.go_id, go_name=entry.name, namespace=entry.namespace,
                    match_type='go_id', score=1.0,
                ))

        tokens = self._token_spans(text)
        token_texts = [t[0] for t in tokens]

        for n in range(self.max_ngram, 0, -1):
            for i, gram in enumerate(ngrams(token_texts, n)):
                phrase_tokens = list(gram)
                while phrase_tokens and phrase_tokens[0] in self._STOPWORDS:
                    phrase_tokens.pop(0)
                while phrase_tokens and phrase_tokens[-1] in self._STOPWORDS:
                    phrase_tokens.pop()
                if not phrase_tokens:
                    continue

                phrase = ' '.join(phrase_tokens)
                entry = self._name_index.get(phrase)
                if not entry:
                    continue

                start_idx = i
                end_idx = i + n - 1
                start = tokens[start_idx][1]
                end = tokens[end_idx][2]
                if any(s <= start < e or s < end <= e for s, e in matched_ranges):
                    continue

                match_type = 'exact' if phrase == entry.name.lower() else 'synonym'
                results.append(NLTKNERResult(
                    span=text[start:end], start=start, end=end,
                    go_id=entry.go_id, go_name=entry.name, namespace=entry.namespace,
                    match_type=match_type, score=1.0,
                ))
                matched_ranges.add((start, end))

        results.sort(key=lambda r: r.start)
        seen = set()
        unique: List[NLTKNERResult] = []
        for r in results:
            key = (r.start, r.end, r.go_id)
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique

    def normalize(self, term: str) -> Optional[GOEntry]:
        return self._name_index.get(term.lower().strip())

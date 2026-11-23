"""Minimal stdlib HTML helpers for Clash Royale Fandom wiki pages.

parse_tables(html) -> list of tables; each table is a dict with
  'id', 'class', 'caption_context' (nearest preceding heading/subheader text),
  'rows' (list of list of cell strings; header cells included).
"""
from html.parser import HTMLParser
import re


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables = []      # finished tables
        self.stack = []       # open tables (nested tables possible)
        self.cell = None      # list of text pieces of the open cell
        self.row = None
        self.context = ""     # last heading-ish text outside tables
        self._ctx_buf = None
        self._ctx_tag = None
        self.skip = 0         # inside <style>/<script>/<sup class=reference>

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("style", "script"):
            self.skip += 1
            return
        if tag == "table":
            self.stack.append({"id": a.get("id", ""), "class": a.get("class", ""),
                               "context": self.context, "rows": []})
            return
        if not self.stack:
            if tag in ("h2", "h3", "h4") or (tag in ("div", "span", "th") and "subheader" in (a.get("class") or "").lower()):
                self._ctx_buf = []
                self._ctx_tag = tag
            return
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.cell = []
            self.cell_span = int(re.sub(r"\D", "", a.get("colspan", "1")) or 1)
        elif tag == "br" and self.cell is not None:
            self.cell.append(" ")
        elif tag == "img" and self.cell is not None:
            alt = a.get("alt") or ""
            if alt and not self.cell:
                pass

    def handle_endtag(self, tag):
        if tag in ("style", "script"):
            self.skip = max(0, self.skip - 1)
            return
        if tag == "table" and self.stack:
            t = self.stack.pop()
            if self.row:
                t["rows"].append(self.row)
            self.tables.append(t)
            self.row = None
            return
        if not self.stack:
            if self._ctx_buf is not None and tag == self._ctx_tag:
                txt = " ".join("".join(self._ctx_buf).split())
                if txt:
                    self.context = txt
                self._ctx_buf = None
            return
        if tag in ("td", "th") and self.cell is not None and self.row is not None:
            txt = " ".join("".join(self.cell).split())
            self.row.append(txt)
            for _ in range(self.cell_span - 1):
                self.row.append(txt)
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.row:
                self.stack[-1]["rows"].append(self.row)
            self.row = None

    def handle_data(self, data):
        if self.skip:
            return
        if self.cell is not None:
            self.cell.append(data)
        elif not self.stack and self._ctx_buf is not None:
            self._ctx_buf.append(data)


def parse_tables(html):
    p = _TableParser()
    p.feed(html)
    return p.tables


class _TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("style", "script"):
            self.skip += 1
        if tag in ("p", "li", "br", "tr", "h2", "h3", "h4", "div"):
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in ("style", "script"):
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def html_text(html):
    p = _TextParser()
    p.feed(html)
    txt = "".join(p.out)
    return re.sub(r"\n\s*\n+", "\n", txt)

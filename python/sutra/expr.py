"""Alpha expression language: parse `rank(-ts_corr(close, volume, 10))` and evaluate it.

Grammar:
    expr  := term (('+' | '-') term)*
    term  := unary (('*' | '/') unary)*
    unary := '-' unary | atom
    atom  := NUMBER | NAME '(' expr (',' expr)* ')' | NAME | '(' expr ')'

The AST is plain tuples so it hashes (handy for caching sub-expressions and
for de-duplicating candidates in the GP search):
    ('num', 10.0) | ('field', 'close') | ('call', 'rank', (args...)) | ('bin', '+', a, b) | ('neg', a)
"""
import re

import numpy as np

from .ops import OPS

_TOKEN = re.compile(r"\s*(?:(\d+\.?\d*)|([A-Za-z_]\w*)|(.))")


def tokenize(src):
    toks = []
    for num, name, sym in _TOKEN.findall(src):
        if num:
            toks.append(("num", float(num)))
        elif name:
            toks.append(("name", name))
        elif sym.strip():
            toks.append(("sym", sym))
    return toks


class _Parser:
    def __init__(self, toks):
        self.toks, self.i = toks, 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def take(self, sym=None):
        tok = self.peek()
        if sym is not None and tok != ("sym", sym):
            raise SyntaxError(f"expected {sym!r}, got {tok[1]!r} at token {self.i}")
        self.i += 1
        return tok

    def expr(self):
        node = self.term()
        while self.peek() in (("sym", "+"), ("sym", "-")):
            op = self.take()[1]
            node = ("bin", op, node, self.term())
        return node

    def term(self):
        node = self.unary()
        while self.peek() in (("sym", "*"), ("sym", "/")):
            op = self.take()[1]
            node = ("bin", op, node, self.unary())
        return node

    def unary(self):
        if self.peek() == ("sym", "-"):
            self.take()
            return ("neg", self.unary())
        return self.atom()

    def atom(self):
        kind, val = self.take()
        if kind == "num":
            return ("num", val)
        if kind == "sym" and val == "(":
            node = self.expr()
            self.take(")")
            return node
        if kind == "name":
            if self.peek() == ("sym", "("):
                self.take("(")
                args = [self.expr()]
                while self.peek() == ("sym", ","):
                    self.take()
                    args.append(self.expr())
                self.take(")")
                if val not in OPS:
                    raise SyntaxError(f"unknown operator {val!r}")
                return ("call", val, tuple(args))
            return ("field", val)
        raise SyntaxError(f"unexpected token {val!r}")


def parse(src):
    p = _Parser(tokenize(src))
    node = p.expr()
    if p.i != len(p.toks):
        raise SyntaxError(f"trailing input at token {p.i}")
    return node


_BIN = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide}


def evaluate(node, fields, cache=None):
    """fields: dict name -> (T, N) panel (plus 'sector' -> (N,) ints if used).
    cache: optional dict; identical sub-trees are computed once."""
    if cache is not None and node in cache:
        return cache[node]
    kind = node[0]
    if kind == "num":
        out = node[1]
    elif kind == "field":
        out = fields[node[1]]
    elif kind == "neg":
        out = -evaluate(node[1], fields, cache)
    elif kind == "bin":
        with np.errstate(invalid="ignore", divide="ignore"):
            out = _BIN[node[1]](evaluate(node[2], fields, cache), evaluate(node[3], fields, cache))
    elif kind == "call":
        name, args = node[1], node[2]
        if name == "group_neutralize":
            out = OPS[name](evaluate(args[0], fields, cache), fields["sector"])
        else:
            vals = [int(a[1]) if a[0] == "num" else evaluate(a, fields, cache) for a in args]
            out = OPS[name](*vals)
    else:
        raise ValueError(node)
    if cache is not None:
        cache[node] = out
    return out

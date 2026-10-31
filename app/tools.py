"""The agent's tools. Arguments arrive already validated (app/schemas.py); this module
only runs allow-listed tools and formats their output for the model.

Tool output is untrusted data (it is scraped web text), so it is fenced and the
system prompt tells the model never to follow instructions inside it.
"""
import ast
import operator

from app import config
from app.retrieve import Hit, Index
from app.schemas import CalculateAction, SearchAction

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def calculate(expression: str) -> float:
    """A sandboxed calculator: parses arithmetic into an AST and evaluates only number
    literals and + - * / // % ** — no names, calls or attributes, so no eval()."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](left, right)
        raise ValueError(f"unsupported expression element: {type(node).__name__}")

    if len(expression) > 200:
        raise ValueError("expression too long")
    result = ev(ast.parse(expression, mode="eval"))
    return round(result, 4) if isinstance(result, float) else result


def format_hits(hits: list[Hit]) -> str:
    if not hits:
        return "(no results above the relevance floor — try other keywords, or answer that the guides don't cover it)"
    return "\n\n".join(f"[{h.chunk.id}] ({h.chunk.destination} › {h.chunk.section})\n{h.chunk.text}" for h in hits)


def run_search(index: Index, args: SearchAction) -> tuple[str, list[Hit]]:
    hits = index.search(args.query, k=config.TOP_K, min_score=config.MIN_SCORE, destination=args.destination)
    return f"<search_results>\n{format_hits(hits)}\n</search_results>", hits


def run_calculate(args: CalculateAction) -> str:
    try:
        return f"<calculator>{args.expression} = {calculate(args.expression)}</calculator>"
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as e:
        return f"<calculator>error: {e}</calculator>"

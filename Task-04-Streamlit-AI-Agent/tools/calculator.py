
import ast
import math
import operator

from langchain_core.tools import tool

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_FUNCS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
}

_CONSTS = {"pi": math.pi, "e": math.e}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]

    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("Exponent too large")

        return _OPS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))

    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
    ):
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])

    raise ValueError("Unsupported expression")


def _format_result(result):
    if isinstance(result, float):
        if not math.isfinite(result):
            raise ValueError("Result is not finite")

        if result.is_integer():
            return str(int(result))

        return format(result, ".12g")

    return str(result)


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Supports arithmetic operators, parentheses, mathematical functions,
    and constants pi and e.
    Example: '(12500 + 8300) * 0.15'
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval(tree)
        return _format_result(result)

    except ZeroDivisionError:
        return "Error: division by zero"

    except Exception as exc:
        return f"Error: could not evaluate '{expression}' ({exc})"

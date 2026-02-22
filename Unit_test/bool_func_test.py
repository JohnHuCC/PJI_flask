from sympy.abc import a, b, c, d, e, f, g, h, i, j
from sympy.logic import simplify_logic
from pyeda.boolalg.expr import exprvar
from pyeda.boolalg.expr import expr, OrOp, AndOp
from pyeda.boolalg import boolfunc
from pyeda.boolalg.minimization import espresso_exprs
from sympy.logic.boolalg import to_dnf
import warnings
warnings.filterwarnings("ignore")


def parse_ast(e):
    if isinstance(e, OrOp):
        return " | ".join([parse_ast(i) for i in e.xs])
    elif isinstance(e, AndOp):
        return " & ".join([parse_ast(i) for i in e.xs])
    else:
        return "{}".format(e)


a, b, c, d, e, f, g, h, i, j = map(
    exprvar, ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
expr = (a & b & ~c) | (~a & b & ~c) | (
    f & g & ~h) | (f & ~g & h) | (~i & j) | (~j & i) | (d & e) | (~d & e & i)
f2 = ~a & ~b & ~c | ~a & ~b & c | a & ~b & c | a & b & c | a & b & ~c
espresso_expr, f2m = espresso_exprs(expr.to_dnf(), f2.to_dnf())

simplified_expr = simplify_logic(expr, force=True)
espresso_simplified_expr = parse_ast(espresso_expr)
print('Kmap:', simplified_expr)
print('Espresso:', espresso_simplified_expr)

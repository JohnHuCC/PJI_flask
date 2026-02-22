from pyeda.boolalg import expr

# 原始表达式
original_expr = expr.Or(
    expr.Variable("A"), expr.Variable("B"), expr.Variable("C")
)

# 简化结果表达式
simplified_expr = expr.Or(
    expr.Variable("A"), expr.Variable("B")
)

# 使用 simplify 函数进行简化
# simplified_result = expr.simplify(simplified_expr)

# 验证结果
is_result_equal = (original_expr.equivalent(simplified_expr))

# 打印结果
print("验证结果：", is_result_equal)

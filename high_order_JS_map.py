# %%

import pytest

class Calculator:
    def add(self, a, b):
        return a + b
    
    def divide(self, a, b):
        if b == 0:
            raise ZeroDivisionError("不能除以零")
        return a / b


# %%


# 简单测试
def test_add():
    calc = Calculator()
    assert calc.add(2, 3) == 5

# 参数化测试
@pytest.mark.parametrize("a,b,expected", [
    (2, 3, 5),
    (-1, 1, 0),
    (0, 0, 0),
    (100, 200, 300)
])
def test_add_multiple_cases(a, b, expected):
    calc = Calculator()
    assert calc.add(a, b) == expected

# 异常测试
def test_divide_by_zero():
    calc = Calculator()
    with pytest.raises(ZeroDivisionError):
        calc.divide(10, 0)

# 夹具(Fixture)
@pytest.fixture
def calculator():
    """提供一个Calculator实例"""
    return Calculator()

def test_with_fixture(calculator):
    assert calculator.add(1, 2) == 3
# %%

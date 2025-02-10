# Copyright © 2024 Giovanni Squillero <giovanni.squillero@polito.it>
# https://github.com/squillero/computational-intelligence
# Free under certain conditions — see the license for details.

import numpy as np


def f(x: np.ndarray) -> np.ndarray:
    return x[0] - (-0.188 * x[1])

# 20250802_0052
def f1(x: np.ndarray) -> np.ndarray: 
    return np.sin(x[0])


def f2(x: np.ndarray) -> np.ndarray:
    return (((((-(-8.583) + -7.783) + ((x[1] - -((5.431 * 4.621))) - (-7.014 - 1.482))) * (0.882 - -((((-8.583 + ((-(-8.583) + (((np.cos(0.844) * (((0.882 - 1.482) - 1.482) * (np.cos((-1.876 + -(0.283))) * 5.500))) + np.abs((-1.876 + -(0.283)))) + np.sin(0.844))) + ((0.882 + 5.431) - x[0]))) + 0.882) * 4.621)))) - (np.abs((-((np.abs((((3.810 - (-(-8.583) * (x[1] - -(((-7.548 + 0.882) * 4.621))))) * 9.486) * -1.692)) + -3.815)) - x[1])) * (-9.234 + ((-(-8.583) + -7.783) + ((0.882 + -(-8.583)) - x[0]))))) * ((((-1.876 + (((np.cos(0.844) * (((0.882 - 0.844) - 1.482) * (np.cos((-9.234 + ((-(-8.583) + (((np.cos(0.844) * (((0.882 - 1.482) - 1.482) * (np.cos((-1.876 + -(0.283))) * 5.500))) + np.abs((-1.876 + -(0.283)))) + np.sin(0.844))) + ((0.882 + 5.431) - x[0])))) * 5.500))) + np.abs(-7.548)) + np.sin(0.844))) * ((-7.014 - 1.482) * (np.cos((-1.876 + -(0.283))) * 5.500))) + np.abs(-7.548)) + np.sin(0.844)))


def f3(x: np.ndarray) -> np.ndarray:
    return ((((1.759 + 1.702) * np.abs(x[0])) - (np.abs((1.109 - x[1])) * ((x[1] * 1.693) + 2.000))) * (1.466 + 1.235))


def f4(x: np.ndarray) -> np.ndarray:
    return (-((-(np.cos(np.abs(x[1]))) - -((-(np.cos(np.abs(np.abs(-(x[1]))))) - np.cos(np.abs(x[1])))))) - (-(-((-(np.cos(x[1])) - -((-(np.cos(-1.719)) - -(((-(np.cos(x[1])) - np.cos(np.abs(x[1]))) - -(-1.719)))))))) - -((-(np.cos(x[1])) - -(-1.719)))))

# 20250802_0200
def f5(x: np.ndarray) -> np.ndarray:
    return 0    # 20250802_0215
    #return np.divide(np.cos((np.abs(np.sin(17.980)) - -((np.cos(-8.687) + (np.cos(-8.687) + 17.980))))), (np.abs(np.sin(17.980)) - -((np.cos(np.sin(17.980)) + (np.cos(-8.687) + 17.980)))), where=(np.abs(np.sin(17.980)) - -((np.cos(np.sin(17.980)) + (np.cos(-8.687) + 17.980))))!=0, out=np.array([1000000000.0]))

def f6(x: np.ndarray) -> np.ndarray:
    # 20250902_0200
    #return (((x[1] + np.cos((-10.707 + ((x[1] + ((((-10.707 + (x[0] / np.abs(x[0]))) + x[0]) + ((-10.707 + ((x[1] + x[0]) / np.abs(np.abs(3.434)))) + (x[1] + x[0]))) + 3.434)) / np.abs(-14.343))))) + np.cos((-10.707 + (x[0] / np.abs(3.434))))) + ((x[1] + np.cos((-10.707 + ((x[1] + x[0]) / np.abs(np.abs(3.434)))))) + np.cos((-10.707 + (x[0] / np.abs(3.434))))))
    return (((x[1] - ((((-1.140 * np.sin(np.cos(-0.869))) * x[1]) + np.cos(np.sin(np.sin(-1.230)))) - -0.499)) + ((np.abs(np.sin(np.abs(np.sin(1.844)))) + -(-1.140)) - (np.cos(np.sin(np.sin(-1.230))) * x[0]))) - np.abs(np.abs(np.sin(np.sin(np.abs(np.abs(-(-1.140))))))))

def f7(x: np.ndarray) -> np.ndarray:
    return np.abs(((-(1.935) - (0.215 - -((x[1] * x[0])))) * (np.cosh((x[1] * x[0])) - ((x[1] * x[0]) * -1.424))))


def f8(x: np.ndarray) -> np.ndarray:
    return ((((x[5] + np.sinh(x[5])) + np.sinh(x[5])) + np.sinh((-1.283 - x[5]))) * np.sinh((0.597 - x[5])))

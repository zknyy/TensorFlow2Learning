#  Copyright (c) 2020. by Eric. All rights reserved.
#  本项目中的所有代码和内容遵循 GNU自由文档许可证1.3或更高版本下发布，如果用于任何商业用途都需经本人同意。任何转载都请注明出处。
#  Email:  [liangzp2k#hotmail.com]
#
#
# 房价问题，线性回归， NumPy 下的线性回归
# 考虑一个实际问题，某城市在 2013 年 - 2017 年的房价如下表所示：
# 年份     2013    2014    2015    2016    2017
# 房价     12000   14000   15000   16500   17500

import numpy as np


X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

# 归一化
X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

print(X, y)  # [0.   0.25 0.5  0.75 1.  ] [0.         0.36363637 0.54545456 0.8181818  1.        ]

a, b = 0, 0

num_epoch = 10000
learning_rate = 5e-4
for e in range(num_epoch):
    # 手动计算损失函数关于自变量（模型参数）的梯度
    y_pred = a * X + b
    grad_a, grad_b = 2 * (y_pred - y).dot(X), 2 * (y_pred - y).sum()

    # 更新参数
    a, b = a - learning_rate * grad_a, b - learning_rate * grad_b

print(a, b)

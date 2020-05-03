#  Copyright (c) 2020. by Eric. All rights reserved.
#  本项目中的所有代码和内容遵循 GNU自由文档许可证1.3或更高版本下发布，如果用于任何商业用途都需经本人同意。任何转载都请注明出处。
#  Email:  [liangzp2k#hotmail.com]
#
#
# 房价问题，线性回归， Tensorflow 下的线性回归
# 考虑一个实际问题，某城市在 2013 年 - 2017 年的房价如下表所示：
# 年份     2013    2014    2015    2016    2017
# 房价     12000   14000   15000   16500   17500

import tensorflow as tf
import numpy as np

X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

# 归一化
X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

print(X, y)  # [0.   0.25 0.5  0.75 1.  ] [0.         0.36363637 0.54545456 0.8181818  1.        ]

X = tf.constant(X)
y = tf.constant(y)

a = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)
variables = [a, b]

num_epoch = 10000
optimizer = tf.keras.optimizers.SGD(learning_rate=5e-4)
for e in range(num_epoch):
    # 使用tf.GradientTape()记录损失函数的梯度信息
    with tf.GradientTape() as tape:
        y_pred = a * X + b
        loss = tf.reduce_sum(tf.square(y_pred - y))
    # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
    grads = tape.gradient(loss, variables)
    # TensorFlow自动根据梯度更新参数
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

print(a, b)
#  <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=0.97637>
#  <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=0.057565063>

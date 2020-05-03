#  Copyright (c) 2020. by Eric. All rights reserved.
#  本项目中的所有代码和内容遵循 GNU自由文档许可证1.3或更高版本下发布，如果用于任何商业用途都需经本人同意。任何转载都请注明出处。
#  Email:  [liangzp2k#hotmail.com]
#
#  对函数 y = 7*x^2  (x=3) 求导  (自动求导机制)

import tensorflow as tf

x = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:  # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
    y = 7 * tf.square(x)
y_grad = tape.gradient(y, x)  # 计算y关于x的导数
print([y, y_grad]) # 当x=3时，求 y 和 dy 的值，分别为 63 和 42
# 输出
# [<tf.Tensor: shape=(), dtype=float32, numpy=63.0>, <tf.Tensor: shape=(), dtype=float32, numpy=42.0>]
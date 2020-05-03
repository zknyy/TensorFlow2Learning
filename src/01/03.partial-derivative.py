#  Copyright (c) 2020. by Eric. All rights reserved.
#  本项目中的所有代码和内容遵循 GNU自由文档许可证1.3或更高版本下发布，如果用于任何商业用途都需经本人同意。任何转载都请注明出处。
#  Email:  [liangzp2k#hotmail.com]
#
#  求偏导数 (自动求导机制)


import tensorflow as tf

X = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[1.], [2.]])
w = tf.Variable(initial_value=[[1.], [2.]])
b = tf.Variable(initial_value=1.)
with tf.GradientTape() as tape:
    L = tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))
w_grad, b_grad = tape.gradient(L, [w, b])  # 计算L(w, b)关于w, b的偏导数
print(L, w_grad, b_grad)
#输出
#tf.Tensor(125.0, shape=(), dtype=float32) tf.Tensor(
#[[ 70.]
# [100.]], shape=(2, 1), dtype=float32) tf.Tensor(30.0, shape=(), dtype=float32)
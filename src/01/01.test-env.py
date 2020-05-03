#  Copyright (c) 2020. by Eric. All rights reserved.
#  本项目中的所有代码和内容遵循 GNU自由文档许可证1.3或更高版本下发布，如果用于任何商业用途都需经本人同意。任何转载都请注明出处。
#  Email:  [liangzp2k#hotmail.com]

#  用于测试tf2环境的可用性
import tensorflow as tf

A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])
C = tf.matmul(A, B)

print(C)

print (tf.__version__)
# 输出如下结果为正常
# tf.Tensor(
# [[19 22]
#  [43 50]], shape=(2, 2), dtype=int32)
# 2.1.0
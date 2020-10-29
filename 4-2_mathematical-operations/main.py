import tensorflow as tf
import numpy as np


def scalar_operations():
    a = tf.constant([[1.0, 2], [-3, 4.0]])
    b = tf.constant([[4.0, 6], [7.0, 8.0]])
    # Operator overloading
    tf.print(a + b)
    tf.print(a - b)
    tf.print(a * b)
    tf.print(a / b)
    tf.print(a ** (0.5))
    tf.print(a % 3)
    tf.print(a // 3)
    tf.print(a >= 2)
    tf.print((a >= 2) & (a <= 3))
    tf.print((a >= 2) | (a <= 3))
    tf.print(a == 5)
    tf.print(tf.sqrt(a))

    a = tf.constant([1.0, 8.0])
    b = tf.constant([5.0, 6.0])
    c = tf.constant([6.0, 7.0])
    tf.print(tf.add_n([a, b, c]))
    tf.print(tf.maximum(a, b))
    tf.print(tf.minimum(a, b))

    x = tf.constant([0.9, -0.8, 100.0, -20.0, 0.7])
    y = tf.clip_by_value(x, clip_value_min=-1, clip_value_max=1)
    z = tf.clip_by_norm(x, clip_norm=3)
    tf.print(x)
    tf.print(y)
    tf.print(z)


def vector_operations():
    a = tf.range(1, 10)
    tf.print(a)
    tf.print(tf.reduce_sum(a))
    tf.print(tf.reduce_mean(a))
    tf.print(tf.reduce_max(a))
    tf.print(tf.reduce_min(a))
    tf.print(tf.reduce_prod(a))

    b = tf.reshape(a, (3, 3))
    tf.print(b)
    tf.print(tf.reduce_sum(b, axis=1, keepdims=True))
    tf.print(tf.reduce_sum(b, axis=0, keepdims=True))

    p = tf.constant([True, False, False])
    q = tf.constant([False, False, True])
    tf.print(tf.reduce_all(p))
    tf.print(tf.reduce_any(q))

    # foldright
    s = tf.foldr(lambda a, b: a + b, tf.range(10))
    tf.print(s)

    a = tf.range(1, 10)
    tf.print(tf.math.cumsum(a))
    tf.print(tf.math.cumprod(a))

    a = tf.range(1, 10)
    tf.print(tf.argmax(a))
    tf.print(tf.argmin(a))

    # Sorting
    a = tf.constant([1, 3, 7, 5, 4, 8])
    values, indices = tf.math.top_k(a, 3, sorted=True)
    tf.print(values)
    tf.print(indices)


def matrix_operations():
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[2, 0], [0, 2]])
    tf.print(a)
    tf.print(b)
    tf.print(a @ b)

    a = tf.constant([[1.0, 2], [3, 4]])
    tf.print(tf.transpose(a))

    a = tf.constant([[1.0, 2], [3.0, 4]], dtype=tf.float32)
    # inv requires floats
    tf.print(tf.linalg.inv(a))

    a = tf.constant([[1.0, 2], [3, 4]])
    tf.print(tf.linalg.trace(a))

    a = tf.constant([[1.0, 2], [3, 4]])
    tf.print(tf.linalg.det(a))

    a = tf.constant([[1.0, 2], [5, 4]])
    tf.print(tf.linalg.eigvals(a))

    a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    q, r = tf.linalg.qr(a)
    tf.print(a)
    tf.print(q)
    tf.print(r)
    tf.print(q @ r)

    a = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=tf.float32)
    s, u, v = tf.linalg.svd(a)
    tf.print("a", a)
    tf.print("u", u)
    tf.print("s", s)
    tf.print("v", v)
    tf.print(u @ tf.linalg.diag(s) @ tf.transpose(v))


def broadcasting_rules():
    a = tf.constant([1, 2, 3])
    b = tf.constant([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    tf.print(b + a)
    tf.print(tf.broadcast_to(a, b.shape))
    tf.print(tf.broadcast_static_shape(a.shape, b.shape))

    c = tf.constant([1, 2, 3])
    d = tf.constant([[1], [2], [3]])
    tf.print(tf.broadcast_dynamic_shape(tf.shape(c), tf.shape(d)))
    tf.print(c + d)


if __name__ == "__main__":
    scalar_operations()
    vector_operations()
    matrix_operations()
    broadcasting_rules()

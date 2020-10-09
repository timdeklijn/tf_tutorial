import numpy as np
import tensorflow as tf

# Constants ============================================================================

# Tensor Types
i = tf.constant(1)  # tf.int32
l = tf.constant(1, dtype=tf.int64)
f = tf.constant(1.23)  # tf.float32
d = tf.constant(3.14, dtype=tf.double)
s = tf.constant("hello world")
tf.constant(True)

print("--Types")
print("tf.int64 == np.int64", tf.int64 == np.int64)
print("tf.bool == np.bool", tf.bool == np.bool)
print("tf.double == np.float64", tf.double == np.float64)
print("tf.string == np.unicode", tf.string == np.unicode)  # not equal
print("\n")


# Tensor Ranks
print("--Ranks")
scalar = tf.constant(True)  # scalar value, tensor of rank 0
print("Scalar Rank       :", tf.rank(scalar).numpy())
print("Scalar Dimensions :", scalar.numpy().ndim)

vector = tf.constant([1.0, 2.0, 3.0, 4.0])  # Vector, rank 1
print("Vector Rank       :", tf.rank(vector).numpy())
print("Vector Dimensions :", vector.numpy().ndim)

matrix = tf.constant([[1.0, 2.0], [3.0, 4.0]])  # matrix, rank 2
print("Matrix Rank       :", tf.rank(matrix).numpy())
print("Matrix Dimensions :", matrix.numpy().ndim)

tensor3 = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
print("Tensor3 Rank      :", tf.rank(tensor3).numpy())
print("Tensor3 Dimensions:", tensor3.numpy().ndim)

tensor4 = tf.constant(
    [
        [[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]],
        [[[5.0, 5.0], [6.0, 6.0]], [[7.0, 7.0], [8.0, 8.0]]],
    ]
)
print("Tensor4 Rank      :", tf.rank(tensor4).numpy())
print("Tensor4 Dimensions:", tensor4.numpy().ndim)
print("\n")


# Cast types
print("-- Cast types")
h = tf.constant([123, 456], dtype=tf.int32)
f = tf.cast(h, tf.float32)
print(h, f)
print("h.dtype, f.dtype", h.dtype, f.dtype)

# Convert to numpy
y = tf.constant([[1.0, 2.0], [3.0, 4.0]])
print(y)
print("y.numpy()", y.numpy())
print("y.shape", y.shape)

# Strings
u = tf.constant(u"Hello World")
print(u)
print("u.numpy()", u.numpy())
print("u.numpy().decode('utf-8')", u.numpy().decode("utf-8"))
print("\n")

# Variable Tensors =====================================================================

# Calculations on constants will create new constants
print("-- Variables")
c = tf.constant([1.0, 2.0])
print("c = tf.constant([1.0, 2.0])")
print(c)
print("id c", id(c))
c = c + tf.constant([1.0, 1.0])
print("c = c + tf.constant([1.0, 1.0])")
print(c)
print("id c", id(c))

#  Variables can be modified
v = tf.Variable([1.0, 2.0], name="v")
print("v = tf.Variable([1.0, 2.0], name='v')")
print(v)
print("id v", id(v))
v.assign_add([1.0, 1.0])
print("v.assign_add([1.0, 1.0])")
print(v)
print("id v", id(v))

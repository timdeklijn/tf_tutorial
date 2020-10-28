import tensorflow as tf
import numpy as np

# Tensor creation ======================================================================
def tensor_creation():
    a = tf.constant([1, 2, 3], dtype=tf.float32)
    tf.print(a)

    b = tf.range(1, 10, delta=2)
    tf.print(b)

    c = tf.linspace(0.0, 2 * 3.14, 100)
    tf.print(c)

    d = tf.zeros([3, 3])
    tf.print(d)

    e = tf.ones([3, 3])
    f = tf.zeros_like(e, dtype=tf.float32)
    tf.print(e)
    tf.print(f)

    g = tf.fill([3, 2], 5)
    tf.print(g)

    tf.random.set_seed(1.0)
    h = tf.random.uniform([5], minval=0, maxval=10)
    tf.print(h)

    i = tf.random.normal([3, 3], mean=0.0, stddev=1.0)
    tf.print(i)

    # Truncate at 2x std
    j = tf.random.truncated_normal((5, 5), mean=0.0, stddev=1.0, dtype=tf.float32)
    tf.print(j)

    I = tf.eye(3, 3)
    tf.print(I)
    tf.print(" ")
    t = tf.linalg.diag([1, 2, 3])
    tf.print(t)


# Indexing + Slicing ===================================================================
def indexing_and_slicing():
    tf.random.set_seed(3)
    t = tf.random.uniform([5, 5], minval=0, maxval=10, dtype=tf.int32)
    tf.print(t)

    # row 0
    tf.print(t[0])
    # last row
    tf.print(t[-1])
    # row 1, column 3
    tf.print(t[1, 3])
    tf.print(t[1][3])

    # row 1 to row 3
    tf.print(t[1:3, :])
    # slice(input, from, to)
    tf.print(tf.slice(t, [1, 0], [3, 5]))

    # Variables can be modified
    x = tf.Variable([[1, 2], [3, 4]], dtype=tf.float32)
    x[1, :].assign(tf.constant([0.0, 0.0]))
    tf.print(x)

    a = tf.random.uniform([3, 3, 3], minval=0, maxval=10, dtype=tf.int32)
    # the following is equal:
    tf.print(a[..., 1])
    tf.print(a[:, :, 1])

    # Now we can do irregular indexing + slicing
    scores = tf.random.uniform((4, 10, 7), minval=0, maxval=100, dtype=tf.int32)
    tf.print(scores)

    # Gather along first axis
    p = tf.gather(scores, [0, 5, 9], axis=1)
    tf.print(scores)

    # gather along first and second axis
    s = tf.gather_nd(scores, indices=[(0, 0), (2, 4), (3, 6)])
    tf.print(s)

    # Can also select by masking
    p = tf.boolean_mask(
        scores,
        [True, False, False, False, False, True, False, False, False, True],
        axis=1,
    )
    tf.print(p)

    # Masks can go deeper
    s = tf.boolean_mask(
        scores,
        [
            [True, False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, False, False],
            [False, False, False, False, True, False, False, False, False, False],
            [False, False, False, False, False, False, True, False, False, False],
        ],
    )
    tf.print(s)

    # Boolean indexing
    c = tf.constant([[-1, 1, -1], [2, 2, -2], [3, -3, 3]], dtype=tf.float32)
    tf.print(c, "\n")
    # The following two are equal
    tf.print(tf.boolean_mask(c, c < 0), "\n")
    tf.print(c[c < 0])

    # tf where:
    c = tf.constant([[-1, 1, -1], [2, 2, -2], [3, -3, 3]], dtype=tf.float32)
    # Select and replace based on condition
    d = tf.where(c < 0, tf.fill(c.shape, np.nan), c)
    tf.print(d)
    # Simply get the indices
    indices = tf.where(c < 0)
    tf.print(indices)

    # new tensor by replacing specific indices
    d = c - tf.scatter_nd([[0, 0], [2, 1]], [c[0, 0], c[2, 1]], c.shape)
    tf.print(d)

    # Use scatter_nd to replace based on index
    # Place only negative values from c in a 0-tensor with c.shape
    indices = tf.where(c < 0)
    tf.print(tf.scatter_nd(indices, tf.gather_nd(c, indices), c.shape))


# Dimension Tranform ===================================================================
def dimension_transform():
    a = tf.random.uniform(shape=[1, 3, 3, 2], minval=0, maxval=255, dtype=tf.int32)
    tf.print(a.shape)
    tf.print(a)

    # Reshape tensor
    b = tf.reshape(a, [3, 6])
    tf.print(b.shape)
    tf.print(b)

    # Reshape back
    c = tf.reshape(b, [1, 3, 3, 2])
    tf.print(c.shape)
    tf.print(c)

    # Squeeze can be used to reduce dimensions with only 1 element
    s = tf.squeeze(a)
    tf.print(s.shape)
    tf.print(s)

    # Expand the dimensions of a tensor along a specific axis
    d = tf.expand_dims(s, axis=0)
    tf.print(d.shape)
    tf.print(d)

    # Transpose swaps dimensions AND changes the order of elements in the tensor/memory
    # [Batch, Height, Width, Channel]
    a = tf.random.uniform([100, 600, 600, 4], minval=0, maxval=255, dtype=tf.int32)
    tf.print(a.shape)

    # [Channel, Height, Width, Batch]
    s = tf.transpose(a, perm=[3, 1, 2, 0])
    tf.print(s.shape)


# Combining + Splitting ================================================================
def combining_splitting():
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    c = tf.constant([[9.0, 10.0], [11.0, 12.0]])
    tf.print(a.shape, b.shape, c.shape)
    tf.print(a)
    tf.print(b)
    tf.print(c)
    tf.print("====")

    # Concat tensors
    d = tf.concat([a, b, c], axis=0)
    tf.print(d.shape)
    tf.print(d)

    # Along an axis
    e = tf.concat([a, b, c], axis=1)
    tf.print(d.shape)
    tf.print(e)

    # Stack tensors
    f = tf.stack([a, b, c])
    tf.print(f.shape)
    tf.print(f)

    # Stack along axis=1
    g = tf.stack([a, b, c], axis=1)
    tf.print(g.shape)
    tf.print(g)

    # Split:
    tf.print(d.shape)
    tf.print(d)

    # tf.split(tensor, num or size, axis)
    h = tf.split(d, 3, axis=0)
    tf.print(h)

    # Back to a, b, c (in a list)
    i = tf.split(d, [2, 2, 2], axis=0)
    print(i)


if __name__ == "__main__":
    tensor_creation()
    indexing_and_slicing()
    dimension_transform()
    combining_splitting()

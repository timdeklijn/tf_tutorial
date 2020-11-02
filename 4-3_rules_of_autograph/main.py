import numpy as np
import tensorflow as tf

# ======================================================================================
# Rule 1: Try to us tf.function decorator + tf functions (like tf.print) as
# much as possible
# ======================================================================================

# BAD
@tf.function
def np_random():
    a = np.random.randn(3, 3)
    tf.print(a)


# GOOD
@tf.function
def tf_random():
    a = tf.random.normal((3, 3))
    tf.print(a)


# ======================================================================================
# Rule 2: Avoid defining tf.Variable inside tf.functions
# ======================================================================================


# BAD
@tf.function
def inner_var():
    x = tf.Variable(1.0, dtype=tf.float32)
    x.assign_add(1.0)
    tf.print(x)
    return x


# GOOD
x = tf.Variable(1.0, dtype=tf.float32)


@tf.function
def outer_var():
    x.assign_add(1.0)
    tf.print(x)
    return x


outer_var()
outer_var()


# ======================================================================================
# Rule 3: tf.functions cannot modify struct datatypes except for tf types
# ======================================================================================

# BAD - result in unexpected list
tensor_list = []


@tf.function
def append_tensor(x):
    tensor_list.append(x)
    return tensor_list


append_tensor(tf.constant(5.0))
append_tensor(tf.constant(6.0))
print(tensor_list)

# OK - but dos not increase speed
tensor_list = []


def append_tensor(x):
    tensor_list.append(x)
    return tensor_list


append_tensor(tf.constant(5.0))
append_tensor(tf.constant(6.0))
print(tensor_list)

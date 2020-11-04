import tensorflow as tf


# Create an autograpg function
@tf.function(autograph=True)
def myadd(a, b):
    for i in tf.range(3):
        tf.print(i)
    c = a + b
    print("tracing")
    return c


# Function is run
myadd(tf.constant("hello"), tf.constant("world"))
# Graph is NOT recreated, "tracing" is not printed
myadd(tf.constant("good"), tf.constant("morning"))
# New graph because types changed
myadd(tf.constant(1), tf.constant(2))
# No tensor so graph is recreated every time
myadd("hello", "world")
myadd("good", "morning")

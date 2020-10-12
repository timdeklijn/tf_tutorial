import datetime
from pathlib import Path

import tensorflow as tf

# Static graph in tensorflow 2.x =======================================================
# Old style of creating graphs using Sessions.

g = tf.compat.v1.Graph()
with g.as_default():
    x = tf.compat.v1.placeholder(name="x", shape=[], dtype=tf.string)
    y = tf.compat.v1.placeholder(name="y", shape=[], dtype=tf.string)
    z = tf.strings.join([x, y], name="join", separator=" ")

with tf.compat.v1.Session(graph=g) as sess:
    result = sess.run(fetches=z, feed_dict={x: "hello", y: "world"})
    print(result)

# Dynamic graph, (only) in tensorflow 2.x ==============================================
# Just use functions with tensorflow oprations to create dynamic graphs


def strjoin(x, y):
    z = tf.strings.join([x, y], separator=" ")
    tf.print(z)
    return z


result = strjoin(tf.constant("hello"), tf.constant("world"))
print(result)

# Autograph in tensorflow 2.x ==========================================================
# Decorate functions to include them in autograph


@tf.function
def strjoin(x, y):
    z = tf.strings.join([x, y], separator=" ")
    tf.print(z)
    return z


result = strjoin(tf.constant("hello"), tf.constant("world"))
print(result)

# Show autograph in tensorboard ========================================================
# Autographs can be plotted in tensorboard. Set trace_on and export the trace.

stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = str(Path("logs/" + stamp))  # Create srtring to logdir

writer = tf.summary.create_file_writer(logdir)  # will create folder if not exists
tf.summary.trace_on(graph=True, profiler=True)  # to be shown in tensorboard

result = strjoin("hello", "world")  # run graph

# Write to tensorboard
with writer.as_default():
    tf.summary.trace_export(name="autograph", step=0, profiler_outdir=logdir)

import tensorflow as tf
import datetime

# ======================================================================================
# Simple autograph
# ======================================================================================

x = tf.Variable(1.0, dtype=tf.float32)

# Define input type
@tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)])
def add_print(a):
    x.assign_add(a)
    tf.print(x)
    return x


# This works only for floats
add_print(tf.constant(3.0))

# ======================================================================================
# tf.Module helps creating autographs
# ======================================================================================


class DemoModule(tf.Module):
    def __init__(self, init_value=tf.constant(0.0), name=None):
        super(DemoModule, self).__init__(name=name)
        with self.name_scope:
            self.x = tf.Variable(init_value, dtype=tf.float32, trainable=True)

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)])
    def addprint(self, a):
        self.x.assign_add(a)
        tf.print(self.x)
        return self.x


demo = DemoModule(init_value=tf.constant(1.0))
result = demo.addprint(tf.constant(5.0))
print(demo.variables)
print(demo.trainable_variables)
print(demo.submodules)

# A tf.Module object can be saved as a model
tf.saved_model.save(demo, "model/1", signatures={"serving_default": demo.addprint})

# Load a model
demo2 = tf.saved_model.load("model/1")
demo2.addprint(tf.constant(5.0))
# Run: `!saved_model_cli show --dir model/1 --all` to show info on saved model file

# ======================================================================================
# Write graph to tensorboard
# ======================================================================================

stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = f"log/{stamp}"
writer = tf.summary.create_file_writer(logdir)
# To show graph in tensorfboard, start tracing it
tf.summary.trace_on(graph=True, profiler=True)
# Run the graph
demo = DemoModule(init_value=tf.constant(0.0))
result = demo.addprint(tf.constant(4.0))
# Write graph info to log
with writer.as_default():
    tf.summary.trace_export(name="demomodule", step=0, profiler_outdir=logdir)

# ======================================================================================
# alternative way to add to tf.modul
# ======================================================================================

mymodule = tf.Module()
mymodule.x = tf.Variable(0.0)


@tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)])
def addprint(a):
    mymodule.x.assign_add(a)
    tf.print(mymodule.x)
    return mymodule.x


# Add function to module
mymodule.addprint = addprint
print(mymodule.variables)

tf.saved_model.save(
    mymodule, "model/mymodule", signatures={"serving_default": mymodule.addprint}
)
mymodule2 = tf.saved_model.load("model/mymodule")
mymodule2.addprint(tf.constant(5.0))

# ======================================================================================
# tf.Module, tf.keras.Model and tf.keras.layers.Layer
# ======================================================================================

# A model is nothing more the a tf.Module object.

# All the following are subclasses of tf.Module
print(issubclass(tf.keras.Model, tf.Module))
print(issubclass(tf.keras.layers.Layer, tf.Module))
print(issubclass(tf.keras.Model, tf.keras.layers.Layer))

tf.keras.backend.clear_session()
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(3, input_shape=(10,)))
model.add(tf.keras.layers.Dense(2))
model.add(tf.keras.layers.Dense(1))
# Print info on model/tf.Module
print(model.summary())
print(model.variables)

# Freeze variables in layer 0
model.layers[0].trainable = False
print(model.trainable_variables)
print(model.submodules)
print(model.layers)
print(model.name)
print(model.name_scope())
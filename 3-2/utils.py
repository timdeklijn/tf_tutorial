import tensorflow as tf


@tf.function
def printbar():
    today_ts = tf.timestamp() % (24 * 60 * 60)
    hours = tf.cast(today_ts // 3600 + 2, tf.int32) % tf.constant(24)
    minutes = tf.cast((today_ts % 3600) // 60, tf.int32)
    seconds = tf.cast(tf.floor(today_ts % 60), tf.int32)

    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}", m)) == 1:
            return tf.strings.format("0{}", m)
        return tf.strings.format("{}", m)

    timestring = tf.strings.join(
        [timeformat(hours), timeformat(minutes), timeformat(seconds)], separator=":"
    )

    tf.print(timestring + " " + "==========" * 4)

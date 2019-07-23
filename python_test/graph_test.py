import tensorflow as tf

a = tf.constant(1.3, name='const_a')
b = tf.Variable(3.1, name='variable_b')
c = tf.add(a, b, name='addition')
d = tf.multiply(c, a, name='multiply')
e = tf.add(a, 0, name='final_pred_heatmaps_tmp')
for op in tf.get_default_graph().get_operations():
    print(str(op.name))
variable_names = [v.name for v in tf.get_default_graph().as_graph_def().node]
print(variable_names)

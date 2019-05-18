import tensorflow as tf

# import textCNN
#
#
# class CNNConfig(object):
#     """CNN配置参数"""
#     print_per_batch = 100  # 每多少轮输出一次结果
#     save_per_batch = 10  # 每多少轮存入tensorboard
#     sequenceLength = 200
#     batchSize = 128
#     dataSource = './data/ag_news.txt'
#     stopWordSource = './stopwords_en.txt'
#     numClasses = 4
#     rate = 0.8
#
#     embeddingSize = 200
#     vocab_size = 5000
#     numFilters = 128
#     filterSizes = 3
#     dropoutKeepProb = 0.5
#     hidden_dim = 128
#
#     epoches = 10
#     evaluateEvery = 100
#     checkpointEvery = 100
#     learningRate = 0.001
#
#     embedding_dim = 64  # 词向量维度
#     seq_length = 600  # 序列长度
#     num_classes = 10  # 类别数
#     num_filters = 256  # 卷积核数目
#     kernel_size = 5  # 卷积核尺寸
#     vocab_size = 5000  # 词汇表达小
#
#
# class TextCNN(object):
#     def __init__(self, config1):
#         self.config1 = config1
#         self.input_x = tf.placeholder(tf.int32, shape=[None, config1.sequenceLength], name='input_x')
#         self.input_y = tf.placeholder(tf.float32, shape=[None, 1], name='input_y')
#         self.dropoutKeepProb = tf.placeholder(tf.float32, name='dropout')
#         self.global_step = tf.Variable(0, trainable=False, name='global_step')  # 计数器
#         self.cnn()
#
#     def cnn(self):
#         with tf.device('/cpu:0'):
#             wordEmbedding = textCNN.wordEmbedding
#             embedding = tf.Variable(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec"), name="W")
#             embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
#         with tf.name_scope("cnn"):
#             conv = tf.layers.conv1d(embedding_inputs, self.config1.numFilters, self.config1.filterSizes,
#                                     name="conv")
#             gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
#         with tf.name_scope("score"):
#             fc = tf.layers.dense(gmp, self.config1.hidden_dim, name='fc1')
#             fc = tf.contrib.layers.dropout(fc, self.dropoutKeepProb)
#             fc = tf.nn.relu(fc)
#
#             self.logits = tf.layers.dense(fc, self.config1.numClasses, name='fc2')
#             self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)
#         with tf.name_scope("optimize"):
#             cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
#             self.loss = tf.reduce_mean(cross_entropy)
#             self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config1.learningRate)
#             gradsAndVars = self.optimizer.compute_gradients(self.loss)
#             self.optim = self.optimizer.apply_gradients(gradsAndVars, global_step=self.global_step)
#         with tf.name_scope("accuracy"):
#             correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
#             self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


print(tf.argmax([0.2, 0.7, 0.3]))

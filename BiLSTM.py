import json
import os
import sys
import time
from collections import Counter
from datetime import timedelta

import gensim
import numpy as np
import tensorflow as tf
from sklearn import metrics

import data_helper


class TrainingConfig(object):
    # 训练参数
    epoches = 10
    evaluateEvery = 100
    checkpointEvery = 100
    learningRate = 1e-3
    lr_decay = 0.9  # 学习率衰减率 learning rate decay


class ModelConfig(object):
    # 模型参数
    embeddingSize = 200
    vocab_size = 5000
    numFilters = 128
    hiddenSizes = [256, 256]
    dropoutKeepProb = 0.5
    hidden_dim = 128
    l2RegLambda = 0.0


class Config(object):
    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard
    sequenceLength = 50
    batchSize = 16
    dataSource = './data/ag_news.txt'
    stopWordSource = './stopwords_en.txt'
    numClasses = 4
    rate = 0.8
    training = TrainingConfig()
    model = ModelConfig()


class Dataset(object):
    def __init__(self, config):
        self._dataSource = config.dataSource
        self._stopWordSource = config.stopWordSource

        self._sequenceLength = config.sequenceLength
        self._embeddingSize = config.model.embeddingSize
        self._batchSize = config.batchSize
        self._rate = config.rate

        self._stopWordDict = {}
        self.Content = []
        self.Labels = []

        self.wordEmbedding = None

        self._wordToIndex = {}
        self._indexToWord = {}

    def cleanReview(self, content):
        # 数据处理函数
        content = content.replace('#', '').replace('=', '').replace("\\", "").replace("\'", "").replace('/',
                                                                                                        '').replace(
            '"', '').replace(',', '').replace(
            '.', '').replace('?', '').replace('(', '').replace(')', '')
        content = content.strip().split(" ")
        content = [word.lower() for word in content]
        content = " ".join(content)
        return content

    def _readData(self, filePath):
        """

        :param filePath:
        :return:
        """
        labels, s = data_helper.read_file(filePath)
        contents = []
        for i in range(len(labels)):
            content = self.cleanReview(s[i])
            contents.append(content)
        return labels, contents

    def _contentProcess(self, content, sequenceLength, wordToIndex):
        """

        :param content:
        :param sequenceLength:
        :param wordToIndex:
        :return:
        """
        contentVec = np.zeros((sequenceLength))
        sequenceLen = sequenceLength
        # 判断当前的序列是否小于固定序列长度
        content = content.strip().split(" ")
        # print(content[0])
        if len(content) < sequenceLength:
            sequenceLen = len(content)
        for i in range(sequenceLen):
            if content[i] in wordToIndex:
                # print(content[i])
                contentVec[i] = wordToIndex[content[i]]
            else:
                contentVec[i] = wordToIndex["UNK"]
        return contentVec

    def _genData(self, x, y, rate):
        """

        :param x:
        :param y:
        :return:
        """
        contents = []
        labels = []
        # 遍历所有的文本，将文本中的词幢换为index表示
        for i in range(len(x)):
            # print(i, x[i])
            reviewVec = self._contentProcess(x[i], self._sequenceLength, self._wordToIndex)
            contents.append(reviewVec)
            labels.append(y[i])
        trainIndex = int(len(x) * rate)
        trainContents = np.asarray(contents[:trainIndex], dtype="int64")
        trainLabels = np.array(labels[:trainIndex], dtype="float32")
        evalContents = np.asarray(contents[trainIndex:], dtype="int64")
        evalLabels = np.array(labels[trainIndex:], dtype="float32")
        return trainContents, trainLabels, evalContents, evalLabels

    def __readStopWord(self, stopWordPath):
        with open(stopWordPath, 'r') as f:
            stopWords = f.read()
            stopWordList = stopWords.splitlines()
            self.stopWordDict = dict(zip(stopWordList, list(range(len(stopWordList)))))

    def _genVocabulary(self, contents):
        allwords, subWords = [], []
        for content in contents:
            content = content.strip().split(" ")
            for word in content:
                allwords.append(word)
        for word in allwords:
            if word not in self.stopWordDict:
                subWords.append(word)
        # allwords = [word for content in contents for word in content]
        # subWords = [word for word in allwords if word not in self.stopWordDict]
        wordCount = Counter(subWords)
        sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)
        # print(sortWordCount)
        words = [item[0] for item in sortWordCount if item[1] >= 5]
        vocab, wordEmbedding = self._getWordEmbedding(words)
        # print(wordEmbedding)
        self.wordEmbedding = wordEmbedding
        self._wordToIndex = dict(zip(vocab, list(range(len(vocab)))))
        self._indexToWord = dict(zip(list(range(len(vocab))), vocab))
        with open("./data/wordJson/wordToIndex.json", 'w') as f:
            json.dump(self._wordToIndex, f)
        with open("./data/wordJson/indexToWord.json", 'w') as f:
            json.dump(self._indexToWord, f)

    def _getWordEmbedding(self, words):
        wordVec = gensim.models.KeyedVectors.load_word2vec_format("./word2Vec.bin", binary=True)
        vocab = []
        wordEmbedding = []
        # 添加"pad  UNK
        vocab.append("pad")
        vocab.append("UNK")
        wordEmbedding.append(np.zeros(self._embeddingSize))
        wordEmbedding.append(np.random.rand(self._embeddingSize))
        for word in words:
            try:
                vector = wordVec.wv[word]
                vocab.append(word)
                wordEmbedding.append(vector)
            except:
                print(word) + "不存在于词向量中"
        return vocab, np.array(wordEmbedding)

    def dataGen(self):
        self.__readStopWord(self._stopWordSource)
        labels, contents = self._readData(self._dataSource)

        self._genVocabulary(contents)
        trainContents, trainLabels, evalContents, evalLabels = self._genData(contents, labels, self._rate)
        self.trainContents = trainContents
        self.trainLabels = trainLabels
        self.evalContents = evalContents
        self.evalLabels = evalLabels


class BiLSTM(object):
    """
    Text CNN 用于文本分类
    """

    def __init__(self, config, wordEmbedding):
        # 定义模型的输入
        self.inputX = tf.placeholder(tf.int32, [None, config.sequenceLength], name="inputX")
        self.inputY = tf.placeholder(tf.float32, [None, config.numClasses], name="inputY")
        self.global_step = tf.Variable(0, trainable=False, name='global_step')  # 计数器
        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")

        # 定义l2损失
        l2Loss = tf.constant(0.0)

        # 词嵌入层
        with tf.name_scope("embedding"):
            # 利用预训练的词向量初始化词嵌入矩阵
            self.W = tf.Variable(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec"), name="W")
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            self.embeddedWords = tf.nn.embedding_lookup(self.W, self.inputX)

        # 定义两层双向LSTM的模型
        with tf.name_scope("Bi_LSTM"):
            for idx, hiddenSize in enumerate(config.model.hiddenSizes):
                with tf.name_scope("Bi_LSTM" + str(idx)):
                    # 定义前向Lstm
                    lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                        output_keep_prob=self.dropoutKeepProb)
                    # 定义反向Lstm
                    lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                        output_keep_prob=self.dropoutKeepProb)
                    outputs, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell,
                                                                                  self.embeddedWords, dtype=tf.float32,
                                                                                  scope="bi_lstm" + str(idx))
                    self.embeddedWords = tf.concat(outputs, 2)
        finalOutput = self.embeddedWords[:, -1, :]
        outputSize = config.model.hiddenSizes[-1] * 2
        output = tf.reshape(finalOutput, [-1, outputSize])
        with tf.name_scope("output"):
            outputW = tf.get_variable("outputW", shape=[outputSize, 4],
                                      initializer=tf.contrib.layers.xavier_initializer())
            outputB = tf.Variable(tf.constant(0.1, shape=[4]), name="outputB")
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            self.logits = tf.nn.xw_plus_b(output, outputW, outputB, name="predictions")
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        # 计算二元交叉熵损失
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.inputY)
            self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2Loss

        with tf.name_scope("optimize"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.inputY)
            self.loss = tf.reduce_mean(cross_entropy)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=config.training.learningRate)
            gradsAndVars = self.optimizer.compute_gradients(self.loss)
            self.optim = self.optimizer.apply_gradients(gradsAndVars, global_step=self.global_step)

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.inputY, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def nextBatch(x, y, batchSize):
    # perm = np.arange(len(x))
    # np.random.shuffle(perm)
    # x = x[perm]
    # y = y[perm]
    # numBatches = len(x)
    # for i in range(numBatches):
    #     start = i * batchSize
    #     end = start + batchSize
    #     batchX = np.array(x[start:end], dtype="int64")
    #     batchY = np.array(y[start:end], dtype="float32")
    #     yield batchX, batchY
    data_len = len(x)
    num_batch = int((data_len - 1) / batchSize) + 1
    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]
    for i in range(num_batch):
        start_id = i * batchSize
        end_id = min((i + 1) * batchSize, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        cnn.inputX: x_batch,
        cnn.inputY: y_batch,
        cnn.dropoutKeepProb: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    data_len = len(x_)
    batch_eval = nextBatch(x_, y_, 16)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        # print(x_batch)
        # print(y_batch)
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 0.5)
        loss, acc, logit = sess.run([cnn.loss, cnn.acc, cnn.logits], feed_dict=feed_dict)
        # print(logit)
        total_loss += loss * batch_len
        total_acc += acc * batch_len
    return total_loss / data_len, total_acc / data_len


# print(np.array(evalContents).shape)


def train():
    save_dir = 'checkpoints/textcnn'
    save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    tf.summary.scalar("loss", cnn.loss)
    tf.summary.scalar("accuracy", cnn.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)
    print('Training and evaluating……')

    best_acc_val = 0.0
    last_improved = 0
    required_importment = 1000
    flag = False
    for epoch in range(config.training.epoches):
        batch_train = nextBatch(trainContents, trainLabels, config.batchSize)
        print('Epoch:', epoch + 1)
        start = time.time()
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.model.dropoutKeepProb)
            _, global_step, train_summaries, train_loss, train_accuracy = sess.run(
                [cnn.optim, cnn.global_step,
                 merged_summary, cnn.loss,
                 cnn.acc], feed_dict=feed_dict)
            # print(logits)
            if global_step % config.print_per_batch == 0:
                end = time.time()
                val_loss, val_accuracy = evaluate(sess, evalContents, evalLabels)
                writer.add_summary(train_summaries, global_step)

                if val_accuracy > best_acc_val:
                    saver.save(sess, save_path)
                    best_acc_val = val_accuracy
                    last_improved = global_step
                    improved_str = '*'
                else:
                    improved_str = ''
                print(
                    "step: {},train loss: {:.3f}, train accuracy: {:.3f}, val loss: {:.3f}, val accuracy: {:.3f},training speed: {:.3f}sec/batch {}\n".format(
                        global_step, train_loss, train_accuracy, val_loss, val_accuracy,
                        (end - start) / config.print_per_batch, improved_str))
                start = time.time()
            if global_step - last_improved > required_importment:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
        config.training.learningRate *= config.training.lr_decay


# train()


def test():
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=sess, save_path=save_path)
    print("Testing……")
    test_loss, test_accuracy = evaluate(sess, evalContents, evalLabels)
    msg = 'Test LossL{0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(test_loss, test_accuracy))
    batchSize = config.batchSize
    data_len = len(evalContents)
    num_batch = int((data_len - 1) / batchSize) + 1
    y_test_cls = np.argmax(evalLabels, 1)
    y_pred_cls = np.zeros(shape=len(evalContents), dtype=np.int32)
    for i in range(num_batch):
        start_id = i * batchSize
        end_id = min((i + 1) * batchSize, data_len)
        feed_dict = {
            cnn.inputX: evalContents[start_id:end_id],
            cnn.inputY: evalLabels[start_id:end_id],
            cnn.dropoutKeepProb: 1.0
        }
        y_pred_cls[start_id:end_id] = sess.run(cnn.y_pred_cls, feed_dict=feed_dict)
    print("Precision ,Recall and F1-Score……")
    print(metrics.classification_report(y_test_cls, y_pred_cls,
                                        target_names=['class 0', 'class 1', 'class 2', 'class 3']))
    print("Confusion Matrix……")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python run_cnn.py [train / test]""")
    config = Config()
    tf.reset_default_graph()
    data = Dataset(config)
    data.dataGen()
    wordEmbedding = data.wordEmbedding
    print('Configuring CNN model...')
    cnn = BiLSTM(config, wordEmbedding)
    start_time = time.time()
    print("Loading training and validation data……")
    start_time = time.time()
    trainContents = data.trainContents
    trainLabels = data.trainLabels
    evalContents = data.evalContents
    evalLabels = data.evalLabels
    wordEmbedding = data.wordEmbedding
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    print("loading test data……")
    save_dir = 'checkpoints/textcnn'
    save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径
    tensorboard_dir = 'tensorboard/textcnn'
    t1 = time.time()
    if sys.argv[1] == 'train':
        train()
    else:
        test()

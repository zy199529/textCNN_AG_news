import codecs

import numpy as np


def stop_words():  # 读取停用词
    stop_words = []
    with open('stopwords_en.txt', encoding='utf-8') as f:
        line = f.readline()
        while line:
            stop_words.append(line[:-1])
            line = f.readline()
    stop_words = set(stop_words)
    return stop_words


def read_file(filename):
    contents, labels = [], []
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                line = line.rstrip()
                assert len(line.split('\t')) == 2
                label, content = line.split('\t')
                one_hot = np.eye(4, dtype='int64')
                labels.append(one_hot[int(label) - 1])
                contents.append(content)
            except:
                pass
    return labels, contents


def cleanReview(content):
    # 数据处理函数
    content = content.replace('#', '').replace('=', '').replace("\\", "").replace("\'", "").replace('/', '').replace(
        '"', '').replace(',', '').replace(
        '.', '').replace('?', '').replace('(', '').replace(')', '')
    content = content.strip().split(" ")
    content = [word.lower() for word in content]
    content = " ".join(content)
    return content


# train_labels, train_contents = read_file('./data/ag_news.txt')
# print(np.array(train_labels).shape)
# test_labels, test_contents = read_file('./data/ag_news_test.txt')
# # for i in range(len(train_contents)):
# #     train_content = cleanReview(train_contents[i])
# #     print(train_content)
#     # print(train_contents)
#     # label = train_labels[i]
#     # with codecs.open("news.txt", 'w',encoding='utf-8') as fw:
#     #     fw.write(train_labels)
#
# for i in range(len(test_contents)):
#     print(test_contents[i])
#     test_content = cleanReview(test_contents[i])

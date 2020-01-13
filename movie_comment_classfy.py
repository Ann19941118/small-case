#-*- coding: utf-8 -*-
from keras.datasets import imdb
from keras import layers, models
import numpy as np
from keras.preprocessing.sequence import _remove_long_seq
from keras import optimizers

def data_load(path, num_words=10000, skip_top=0,
              maxlen=None, seed=113, start_char=1,
              oov_char=2, index_from=3, **kwargs):
    with np.load(path) as f:
        x_train, labels_train = f['x_train'], f['y_train']
        x_test, labels_test = f['x_test'], f['y_test']

    np.random.seed(seed)
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    labels_train = labels_train[indices]

    indices = np.arange(len(x_test))
    np.random.shuffle(indices)
    x_test = x_test[indices]
    labels_test = labels_test[indices]

    xs = np.concatenate([x_train, x_test])
    labels = np.concatenate([labels_train, labels_test])

    if start_char is not None:
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        xs = [[w + index_from for w in x] for x in xs]

    if maxlen:
        xs, labels = _remove_long_seq(maxlen, xs, labels)
        if not xs:
            raise ValueError('After filtering for sequences shorter than maxlen=' +
                             str(maxlen) + ', no sequence was kept. '
                                           'Increase maxlen.')
    if not num_words:
        num_words = max([max(x) for x in xs])

    if oov_char is not None:
        xs = [[w if (skip_top <= w < num_words) else oov_char for w in x]
              for x in xs]
    else:
        xs = [[w for w in x if skip_top <= w < num_words]
              for x in xs]

    idx = len(x_train)
    x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
    x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])

    return (x_train, y_train), (x_test, y_test)



def vectorize_sequences(sequences, dimension=10000):
    result = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        result[i, sequence] = 1
    return result



if __name__=='__main__':

    path = r'C:\Users\Schwarz\zili\bike_sharing\imdb.npz'
    (train_data, train_label), (test_data, test_label) = data_load(path, num_words=10000)

    # comment 可视化
    word_index = imdb.get_word_index()
    # h = word_index['the']
    # print(h)
    # word_index 中 (the, 1) 意思是单词the 在我字典中的索引位置是 1，需要把形式 (word, index) 转换成 (index, word)
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    #索引减去3,是0，1，2是为padding
    decode_review = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])
    # print(decode_review)

    #现在的train_data是长度不一的评论单词索引，不能把这个整数序列直接放入模型 需要转化成张量，有两种方法进行转化
    '''
    method1: padding 填充使其具有相同的长度，再把列表转化成形状为(samplies, word_indices)的整张张量， 需要用到keras的Embdding层
    method2: 对列表进行one-hot, 将其转化成0和1组成的向量。只有出现的单词对应的编码为1，未出现的单词编码为0
    '''
    #数据向量化, 下面尝试使用第二种方法 one-hot 将整张列表转化成二进制矩阵
    x_train = vectorize_sequences(train_data)
    # print(x_train.shape)
    x_test = vectorize_sequences(test_data)

    # 标签向量化
    y_train = np.asarray(train_label).astype('float32')
    y_test = np.asarray(test_label).astype('float32')
    # print(x_train[0])
    # one-hot之后把矩阵放到模型中

    # build network
    model = models.Sequential()
    model.add(layers.Dense(units=16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=16, activation='relu'))
    model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(units=64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  #为什么是二分类问题只设置一个输出神经元

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    # 或者通过传入一个optimizer类实例
    # model.compile(optimizer=optimizers.RMSprop(lr=0.001),
    #               loss='binary_crossentropy',
    #               metrics=['accuracy'])
    #或者是使用自定义的损失和指标
    # from keras import losses
    # from keras import metrics
    # model.compile(optimizer=optimizers.RMSprop,
    #               loss=losses.binary_crossentropy,
    #               metrics=metrics.binary_accuracy)

    # 留出验证集
    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]

    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]

    #训练模型
    history = model.fit(partial_x_train, partial_y_train,
              epochs=50, batch_size=512,
              validation_data=(x_val, y_val))

    # print(history.history.keys())
    import matplotlib.pyplot as plt

    history_dict = history.history
    acc_values = history_dict['loss']
    val_acc_values = history_dict['val_loss']

    epochs = range(1, len(acc_values)+1)
    
    # 在同一张图上显示两个图像
    plt.plot(epochs, acc_values, 'bo', label="Training loss")
    plt.plot(epochs, val_acc_values, 'b', label='Validation loss')
    plt.title("Training and validation loss")
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()

    plt.show()
    model.save('movie_comment_classifier.h5')











#################################################
#
# v3. 2019.10.28
#  - Added the parameter at OptionParser
#  - Filename
#
# v2. 2019.10.28
#  - Added Precision, Recall, F1score, Confusion Matrix
#
# v1. 2019.10.27
#  - CNN classification core (training, testing)
#  - 3 fold validation (fixed)
#
#################################################

import sklearn.metrics as sk
import tensorflow as tf
import pandas as pd
import numpy as np
import time
from optparse import OptionParser
import sys
import logging
import datetime

if (len(sys.argv) <= 1):
    print("CNN_3FCV.py -h 또는 --help 를 입력해 메뉴얼을 참조하세요.")
    exit()
todaytime = datetime.datetime.now()
parser = OptionParser(usage = "Usage: %prog -d dataset_file [options]")
parser.add_option('-d','--dataset', type=str, dest='Path', action="store", help='Training Data Path (necessary)')
parser.add_option('-n','--outputname', type=str, dest='File_name', default="", action="store", help='Output File Prefix')
parser.add_option('-c','--conv', type=int, dest='Conv_Layer_Number', default=1, action="store", help='Convolution Layer Number')
parser.add_option('-p','--pool', type=int, dest='Max_Pool_Stride_Number', default=2, action="store", help='Max Pooling Layer Number')
parser.add_option('-f','--fcl', type=int, dest='Fully_Connected_Layer_Number', default=2, action="store", help='Fully Connected Layer Number')
parser.add_option('--hidden', type=int, dest='Hidden_Layer_Number', default=1024, action="store", help='Hidden Layer Number')
parser.add_option('--units', type=int, dest='N_Hidden_Units_Value', default=100, action="store", help='N Hidden Units Value')
parser.add_option('-l','--learning', type=float, dest='Learning_Rate', default=0.00001, action="store", help='Learning Rate Value')
parser.add_option('-i','--iters', type=int, dest='Training_Iters', default=2100, action="store", help='Training Iters Value')
parser.add_option('-b','--batch', type=int, dest='Batch_Size', default=50, action="store", help='Batch Size Value')
parser.add_option('--display', type=float, dest='Display_Step', default=10, action="store", help='Display Step')
parser.add_option('--dropout', type=float, dest='Dropout', default=.5, action="store", help='Dropout Value')

options, args = parser.parse_args()

CLnum = options.Conv_Layer_Number
pool_kernel = options.Max_Pool_Stride_Number
FCLnum = options.Fully_Connected_Layer_Number
HLnum = options.Hidden_Layer_Number

# 사용자 - 데이터 패스 입력
path_dataset = options.Path
Original = pd.read_csv(path_dataset, header=None)

# 데이터 엔트리 이름 제거
dataset = Original.drop(0,0)

for i in range(0,dataset.shape[1]-1):
    dataset[i] = dataset[i].astype(float)

    # 음수 값 양수로 변환
    if dataset[i].min() < 0 :
        dataset[i] = dataset[i] - dataset[i].min()

    # 16777216 -> 2^24 24비트 이상일 시 변환.
    dataset[i][dataset[i] > 16777216] = 16777216
    if dataset[i].max() - dataset[i].min() == 0:
        dataset[i] = 255*(dataset[i] - dataset[i].min())
    else :
        dataset[i] = 255*((dataset[i] - dataset[i].min()) / (dataset[i].max() - dataset[i].min()))


# NxN 형태로 변환을 위해 0 값 입력
label_set = dataset[dataset.shape[1]-1]
N_row = 0
for i in range(2,dataset.shape[1]-1):
    if dataset.shape[1] - 1 - i*i <= 0 :
        N_row = i
        break
# print("N_row:",N_row,",Need :",N_row*N_row-dataset.shape[1]+1)
Rest = N_row*N_row-dataset.shape[1]+1
Row_Number = dataset.shape[1]-1
result = 0
if Rest == 0:
    result = dataset.iloc[0:dataset.shape[0]-1,0:dataset.shape[1]-1]
for i in range(0,Rest):
    if i == 0:
        result = dataset.iloc[0:dataset.shape[0]-1,0:dataset.shape[1]-1]
    df = pd.DataFrame(columns=([Row_Number+i]))
    dataset = dataset.iloc[0:dataset.shape[0]-1,0:result.shape[1]-1]
    result = pd.concat([result,df])
    result = result.fillna(0)

#print(result)

# 데이터 프레임 가져와 "lable"종류 확인
label = label_set.unique()
label = label[0:len(label)]

# Legend -> OneHot Vector로 변환.
for i in range(0,len(label)):
    df = pd.DataFrame(columns=([str(label[i])]))
    label_set = pd.concat([label_set,df],axis=0)
    label_set = label_set.fillna(0)
    label_set[label[i]] =label_set[0].isin([label[i]]).astype(int)
label_set = label_set.iloc[0:result.shape[0],1:label_set.shape[1]]
post_dataset = pd.concat([result,label_set],axis=1).astype(float)

#print(post_dataset.shape[1],len(label))

test_33= []; train_66 = []
test_33_1 = []; train_66_1 = []
test_33_2 = []; train_66_2 = []
test_33_3 = []; train_66_3 = []
# "lable" 종류 별로 3 폴드로 나눠 파일을 만든다.
for i in range(0,label.shape[0]):
    print(label[i],"Case Classification")
    classification = post_dataset[post_dataset[label[i]]==1]
    a = classification[0:int(len(post_dataset[post_dataset[label[i]] == 1]) / 3)]
    b = classification[int(len(post_dataset[post_dataset[label[i]] == 1]) / 3):2 * int(len(post_dataset[post_dataset[label[i]] == 1]) / 3)]
    c = classification[2 * int(len(post_dataset[post_dataset[label[i]] == 1]) / 3):]
    test_33_1.append(a)
    test_33_2.append(b)
    test_33_3.append(c)
    train_66_1.append(pd.concat([b,c],axis=0))
    train_66_2.append(pd.concat([a,c],axis=0))
    train_66_3.append(pd.concat([a,b],axis=0))
test_33.append(pd.concat(test_33_1,axis=0))
test_33.append(pd.concat(test_33_2,axis=0))
test_33.append(pd.concat(test_33_3,axis=0))
train_66.append(pd.concat(train_66_1,axis=0))
train_66.append(pd.concat(train_66_2,axis=0))
train_66.append(pd.concat(train_66_3,axis=0))

# Training parameters
learning_rate_ = options.Learning_Rate
training_iters = options.Training_Iters #for w,b 15000
batch_size = options.Batch_Size
display_step = options.Display_Step
dropout = options.Dropout  # dropout, probability to keep units
n_hidden_units = options.N_Hidden_Units_Value  # the number of neural in each hidden layer

keep_prob = tf.compat.v1.placeholder(tf.float32)  # dropout (keep probability)


# Read csv files
def Shuffling_Data(train, test):
    training_data = train
    testing_data = test
    num_examples = len(training_data)
    perm = np.arange(num_examples)
    np.random.shuffle(perm)
    training_data = np.array(training_data)
    training_data = training_data[perm]
    testing_data = np.array(testing_data)
    return training_data, testing_data, num_examples

def Set_Parameters(training_data, label):
    # Network parameters
    n_input = training_data.shape[1] - label.shape[0]      # data input (img shape: 9*9)
    n_classes = label.shape[0]       # total classes
    # Variables
    x = tf.compat.v1.placeholder(tf.float32, [None, n_input])
    y = tf.compat.v1.placeholder(tf.float32, [None, n_classes])
    return n_input, n_classes, x, y

# CNN
# Getting output from CNN
# Convolutional Layer (feature maps' size = 9x9)
def Convolutional_Layer(X, Input, padding):
    output = 32
    W_conv = tf.Variable(tf.random.truncated_normal(shape=[3, 3, Input, output], stddev=1e-4))
    b_conv = tf.Variable(tf.constant(0.1, shape=[output]))
    h_conv = tf.nn.relu(tf.nn.conv2d(X, W_conv, strides=[1, 1, 1, 1], padding=padding) + b_conv)
    return h_conv, output

# Convolutional Layer Number (feature maps' size = 3x3)
def CL_Count(num, input):
    h_conv = 0; output = 0;
    for i in range(1, num+1):
        if i == 1:
            h_conv = input; output = 1;
        h_conv1, output_result = Convolutional_Layer(h_conv, output, 'SAME')
        if i != 1:
            h_conv = h_conv1; output = output_result;
    return h_conv1

# Max-Pooling Layer (feature maps' size = 2x2)
def Max_Pooling_Layer(Input, pool_kernel, stride):
    h_pool = tf.nn.max_pool2d(Input, ksize=[1, pool_kernel, pool_kernel, 1], strides=[1, stride, stride, 1], padding='SAME')
    return h_pool

# Convert feature maps into vectors
def Flatten(h_pool,size):
    h_pool_flat = tf.reshape(h_pool, [-1, size])
    return h_pool_flat

# Fully-Connected Layer
def Fully_Connected_Layer(Input, InputSize, OutputSize):
    W_fc = tf.Variable(tf.random.truncated_normal(shape=[InputSize, OutputSize], stddev=0.04))
    b_fc = tf.Variable(tf.constant(0.0, shape=[OutputSize]))
    h_fc = tf.nn.relu(tf.matmul(Input, W_fc) + b_fc)
    return h_fc, OutputSize

# Fully-Connected Layer Number
def FCL_Count(num, Input_flat,InputSize, OutputSize):
    h_flat = Input_flat
    for i in range(1,num+1):
        h_fc, output = Fully_Connected_Layer(h_flat, InputSize, OutputSize)
        h_flat = h_fc; InputSize = output;
    return h_fc

# Output layer
def Output_Layer(Input, InputSize, n_classes):
    W_fc3 = tf.Variable(tf.random.truncated_normal(shape=[InputSize, n_classes], stddev=1e-4))
    b_fc3 = tf.Variable(tf.constant(0.1, shape=[n_classes]))
    y_conv = tf.nn.softmax(tf.matmul(Input, W_fc3) + b_fc3)
    return y_conv

# Neural Network Model
# pool_kernel
def Net_Model(x, CLnum, pool_kernel, FCLnum, HLnum, dropout, n_classes):
    # Convert input vector into image
    x = tf.reshape(x, shape=[-1, N_row, N_row, 1])

    # Convolution Number(1,2,3)
    h_conv = CL_Count(CLnum, x)

    # Pooling Type(3)
    stride = pool_kernel
    h_pool = Max_Pooling_Layer(h_conv, pool_kernel, stride)
    FC_kernel = round((9-pool_kernel)/stride)+1
    #h_pool_flat = Flatten(h_pool,FC_kernel*FC_kernel*32)
    h_pool_flat = Flatten(h_pool, FC_kernel * FC_kernel * 32)

    # FC Net Layer Number(1, 2), Hidden Layer Number(512, 1024)
    h_fc = FCL_Count(FCLnum, h_pool_flat,FC_kernel*FC_kernel*32, HLnum)
    h_fc_drop = tf.nn.dropout(h_fc, dropout)
    y_conv = Output_Layer(h_fc_drop, HLnum, n_classes)
    return y_conv


# serve data by batches
index_in_epoch = 0
def next_batch(batch_size, num_examples):
    global training_data
    global index_in_epoch

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        training_data = training_data[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return training_data[start:end]

def Loss_function(y, pred):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate_).minimize(cross_entropy)
    return cross_entropy, optimizer

def Evaluation(y, pred):
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

# Launching
def Training(cross_entropy, accuracy, n_input):
    # Initializing the variables
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(training_iters):
        # Training batch
        training_batch = next_batch(batch_size, num_examples)

        # Learning
        sess.run(optimizer, feed_dict={x: training_batch[:, 0:n_input],
                                       y: training_batch[:, N_row*N_row:N_row*N_row+len(label)],
                                       keep_prob: dropout})
        # Calculate training accuracy
        if step % display_step == 0:
            loss, acc = sess.run([cross_entropy, accuracy],
                                 feed_dict={x: training_batch[:, 0:n_input],
                                            y: training_batch[:, N_row*N_row:N_row*N_row+len(label)],
                                            keep_prob: 1.})
            print('> Training step %d: minibatch loss = %f, training accuracy = %f' % (step, loss, acc))

    return sess

def Test(accuracy, pred, sess, n_input):
    # Testing start - For calculating running time
    # Calculate testing accuracy
    batch_size_test = 200
    test_size = int(testing_data.shape[0])

    if(batch_size_test >= test_size):
        batch_size_test = test_size

    num_batch = int(test_size/batch_size_test)
    mod_batch = int(test_size)*1.0/batch_size_test - num_batch
    if(mod_batch>0):
        num_batch += 1

    y_true_t = np.array([], dtype=np.int64)
    y_pred_t = np.array([], dtype=np.int64)

    start = 0
    test_accuracy = 0.0
    for i in range(num_batch):
        end = (i+1)*batch_size_test
        if(end >= test_size):
            end = test_size

        #test_accuracy += sess.run(accuracy, feed_dict={x: testing_data[start:end,0:n_input],
        #                                               y: testing_data[start:end,81:81+len(label)],
        #                                               keep_prob: 1.})

        y_p = tf.argmax(pred, 1)
        t_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x: testing_data[start:end,0:n_input],
                                                       y: testing_data[start:end,N_row*N_row:N_row*N_row+len(label)],
                                                       keep_prob: 1.})

        #print( "TT : ", type(y_pred), " : ", y_pred.dtype )
        #print( "TT : ",type(y_pred_t), " : ", y_pred_t.dtype )

        y_pred_t = np.r_[y_pred_t, y_pred]
        #print('y_pred: %s' % (y_pred_t))

        y_true = np.argmax(testing_data[start:end, N_row*N_row:N_row*N_row + len(label)], 1)
        y_true_t = np.r_[y_true_t, y_true]
        #print('y_true: %s' % (y_true_t))

        test_accuracy += t_accuracy

        start = end

    print('Precision : ', sk.precision_score(y_true_t, y_pred_t, average=None))
    print('Recall : ', sk.recall_score(y_true_t, y_pred_t, average=None))
    print('F1_score : ', sk.f1_score(y_true_t, y_pred_t, average=None))
    print('confusion_matrix')
    print(sk.confusion_matrix(y_true_t, y_pred_t))
    Accuracy = test_accuracy/num_batch
    sess.close()
    return Accuracy

for i in range(0,3):
    f = open(options.File_name+'_'+todaytime.strftime("%Y-%m-%d")+'_output' + str(i+1) + '.txt', 'w')
    sys.stdout = f
    print(i+1,'th test')
    print('Convolutional_Layer_Num: %d, Pool_Type: %d, Fully_Connected_Layer_Num: %d, Hidden_Layer_Num: %d, Training completed' % (CLnum, pool_kernel, FCLnum, HLnum))
    training_data, testing_data, num_examples = Shuffling_Data(train_66[i], test_33[i])
    n_input, n_classes, x, y = Set_Parameters(training_data, label)
    pred = Net_Model(x, CLnum, pool_kernel, FCLnum, HLnum, keep_prob, n_classes)
    cross_entropy, optimizer = Loss_function(y, pred)
    accuracy = Evaluation(y, pred)
    start_time = time.time()
    print(n_input)
    sess = Training(cross_entropy, accuracy, n_input)
    Timeout = time.time() - start_time
    Accuracy = Test(accuracy, pred, sess, n_input)
    print('Testing accuracy: %f' % (Accuracy))
    print('Running-time: %s secs' % (Timeout))
    f.close()

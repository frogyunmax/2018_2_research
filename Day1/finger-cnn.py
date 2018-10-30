import tensorflow as tf
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import array

###학습 데이터 가공
TRAIN_DIR = 'D:/deeplearning/programming/fingerlanguage/traindata'
train_folder_list = array(os.listdir(TRAIN_DIR))

train_input = []
train_label = []

label_encoder = LabelEncoder()  # LabelEncoder Class 호출
integer_encoded = label_encoder.fit_transform(train_folder_list)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

onehot_encoded = onehot_encoder.fit_transform(integer_encoded)


for index in range(len(train_folder_list)):
    path = os.path.join(TRAIN_DIR, train_folder_list[index])
    path = path + '/'
    img_list = os.listdir(path)
    for img in img_list:
        img_path = os.path.join(path, img)
        #print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        #print(img)
        train_input.append([np.array(img)])
        train_label.append([np.array(onehot_encoded[index])])

train_input = np.reshape(train_input, (-1, 784))
train_label = np.reshape(train_label, (-1, 4))
train_input = np.array(train_input).astype(np.float32)
train_label = np.array(train_label).astype(np.float32)
#print(train_label)
np.save("train_data.npy", train_input)
np.save("train_label.npy", train_label)


###테스트 데이터 가공
TEST_DIR = 'D:/deeplearning/programming/fingerlanguage/testdata'
test_folder_list = array(os.listdir(TEST_DIR))

test_input = []
test_label = []

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(test_folder_list)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

for index in range(len(test_folder_list)):
    path = os.path.join(TEST_DIR, test_folder_list[index])
    path = path + '/'
    img_list = os.listdir(path)
    for img in img_list:
        img_path = os.path.join(path, img)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        test_input.append([np.array(img)])
        test_label.append([np.array(onehot_encoded[index])])

test_input = np.reshape(test_input, (-1, 784))
test_label = np.reshape(test_label, (-1, 4))
test_input = np.array(test_input).astype(np.float32)
test_label = np.array(test_label).astype(np.float32)
np.save("test_input.npy", test_input)
np.save("test_label.npy", test_label)

#실제학습 시작
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 4]) #출력값4개
keep_prob = tf.placeholder(tf.float32)

learning_rate = 0.001

W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W3 = tf.Variable(tf.random_normal([7*7*64, 256], stddev = 0.01))
L3 = tf.reshape(L2, [-1, 7*7*64]) #직전 폴링 계층 크기가 7x7x64, 이를 1차원으로 만든다.
L3 = tf.matmul(L3, W3)
L3 = tf.nn.relu(L3) #활성화함수 적용
#L3 = tf.nn.dropout(L3, keep_prob) #과적함 방지
#편향은 일단제거
#b = tf.Variable(tf.random_normal([4]))

W4 = tf.Variable(tf.random_normal([256, 4], stddev = 0.01))
model = tf.matmul(L3, W4) # + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
#평균 엔트로피 함수
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)  # 초기화 실행시키기

batch_size = 100  # 미니배치 사용 (100개)
total_batch = int(len(train_input) / batch_size)

print('시작합니다. 시간이 오래 걸릴 수 있습니다.')
for epoch in range(20):
    total_cost = 0
    for i in range(total_batch):
        start = ((i+1) * batch_size) - batch_size
        end = ((i+1) * batch_size) # 학습할 데이터를 배치 크기만큼 가져온 뒤,
        batch_xs = train_input[start:end]
        batch_ys = train_label[start:end]

        # 입력값 = batch_xs, 출력값 = batch_ys
        #batch_xs = batch_xs.reshape(-1, 28, 28, 1)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
                    # 최적화+손실값 저장, 입력값X와 평가할 때 쓸 실제값 Y를 넣어준다. 과적합방지 비율 = 70%
        total_cost = total_cost + cost_val
    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost = ', '{:3f}'.format(total_cost / total_batch))
print('최적화 완료!')

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accuracy, feed_dict = {X: test_input, Y: test_label}))

import tensorflow as tf
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import array
import re

###학습 데이터 가공
#학습데이터 가져올 위치
TRAIN_DIR = 'D:/deeplearning/programming/fingerlanguage/traindata'
train_folder_list = array(os.listdir(TRAIN_DIR))

#데이터값들은 input 에, 정답은 label에
train_input = []
train_label = []

label_encoder = LabelEncoder()
#폴더들을 인식, 그것들의 이름을 순차적으로 숫자형 데이터로 전환하기
integer_encoded = label_encoder.fit_transform(train_folder_list)

onehot_encoder = OneHotEncoder(sparse=False)
#onehotencoder 쓰기 위해서 integer_encoded 를 2차원배열으로 전환한다.
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
#해당 값을 가지는 index만 값을 1로 하고 나머지는 다 0을 가지는 배열으로 전환한다.
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

#img를 변환시키는 과정
for index in range(len(train_folder_list)):
    path = os.path.join(TRAIN_DIR, train_folder_list[index])
    path = path + '/'
    #경로에 / 를 추가한다.
    img_list = os.listdir(path)
    for img in img_list:
        # 해당 경로 내 폴더들에 대해서 이미지를 하나씩 읽어준다.
        img_path = os.path.join(path, img)
        #print(img_path)
        #이미지를 읽어온다.
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        #print(img)
        #이미지를 numpy형 배열으로 변환해서 train_input에 넣는다.
        train_input.append([np.array(img)])
        #라벨 (정답) 도 마찬가지로 train_label에 넣는다.
        train_label.append([np.array(onehot_encoded[index])])

#input 데이터를 784개 (28x28) 가 있는 배열으로 변환
#-1은 데이터 개수 정확히 모를 때 사용하는 거
#label 데이터도 마찬가지
train_input = np.reshape(train_input, (-1, 784))
train_label = np.reshape(train_label, (-1, 18))
#최종 변환
train_input = np.array(train_input).astype(np.float32)
train_label = np.array(train_label).astype(np.float32)
#print(train_label)
#이거를 npy 파일로 저장 (실제로 해당폴더에 저장됨)
np.save("train_data.npy", train_input)
np.save("train_label.npy", train_label)


###테스트 데이터 가공
#학습 데이터와 같은 원리
TEST_DIR = 'D:/deeplearning/programming/fingerlanguage/tts'

test_input = []
img_list = os.listdir(TEST_DIR)
for img in img_list:
    img_path = os.path.join(TEST_DIR, img)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    test_input.append([np.array(img)])

test_input = np.reshape(test_input, (-1, 784))
test_input = np.array(test_input).astype(np.float32)
np.save("test_input.npy", test_input)

#실제학습 시작
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 18]) #출력값
#추후 드롭아웃 시 사용할 변수 keep_prob
keep_prob = tf.placeholder(tf.float32)

#학습률 설정
learning_rate = 0.001

#컨볼루션 계층 생성하기(3x3) (내장함수사용)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))#3x3 크기 커널 가진 계층, 오른쪽/아래쪽으로 1칸씩 움직이며, 32개 커널
                                                            #랜덤으로 가중치를 넣는다.
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')#padding SAME : 이미지 외각 (테두리) 까지도 조금 더 명확히 평가가능
L1 = tf.nn.relu(L1) #활성화함수로 relu 적용

#풀링 계층 (내장함수사용) (맥스풀링사용)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#두 번째 계층 역시 같은 방법으로
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01)) #L1 풀링 맵을 기본으로 (32개) -> #이번에는 3x3 의 커널 64개로 구성됨
                                                                 #랜덤으로 가중치를 넣는다.
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#10개의 분류를 만들어내는 계층
W3 = tf.Variable(tf.random_normal([7*7*64, 256], stddev = 0.01))
L3 = tf.reshape(L2, [-1, 7*7*64]) #직전 폴링 계층 크기가 7x7x64, 이를 1차원으로 만든다.
L3 = tf.matmul(L3, W3)
L3 = tf.nn.relu(L3) #활성화함수 적용
#L3 = tf.nn.dropout(L3, keep_prob) #과적함 방지
#편향은 일단제거
#b = tf.Variable(tf.random_normal([4]))

W4 = tf.Variable(tf.random_normal([256, 18], stddev = 0.01))
model = tf.matmul(L3, W4) # + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
#평균 엔트로피 함수
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
#optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost) 이거로도 가능..

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)  # 초기화 실행시키기

batch_size = 100  # 미니배치 사용 (100개)
total_batch = int(len(train_input) / batch_size)
all_epoch = 80

print('학습 시작!')
for epoch in range(all_epoch):
    total_cost = 0
    for i in range(total_batch):
        start = ((i+1) * batch_size) - batch_size
        end = ((i+1) * batch_size) # 학습할 데이터를 배치 크기만큼 가져온 뒤,
        batch_xs = train_input[start:end]
        batch_ys = train_label[start:end]
        #batch_xs, batch_ys 정의해주기
        # 입력값 = batch_xs, 출력값 = batch_ys

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
        # 최적화+손실값 저장, 입력값X와 평가할 때 쓸 실제값 Y를 넣어준다. 과적합방지 비율 = 70%
        total_cost = total_cost + cost_val
        #오차값 구하기
    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost = ', '{:3f}'.format(total_cost / total_batch))
print('학습 완료!')

#결과확인
result = sess.run(model, feed_dict = {X: test_input})

#각각 model 에서 얻은 결론값에 해당하는 한글 초성을 출력한다.
#현재 index 0 = ㄱ 1 = ㄹ 2 = ㅍ 3 = ㅎ
char = {0 : 'ㄱ', 1 : 'ㄴ', 2 : 'ㄷ', 3 : 'ㄹ', 4 : 'ㅁ', 5 : 'ㅂ', 6 : 'ㅅ',
        7 : 'ㅇ', 8 : 'ㅈ', 9 : 'ㅊ', 10 : 'ㅋ', 11 : 'ㅌ', 12 : 'ㅍ',  13 : 'ㅎ',
        14 : 'ㅓ', 15 : 'ㅗ', 16 : 'ㅣ', 17 : 'ㅐ'}
for i in range(len(result)):
    max = -100
    ind = 0
    for j in range(0, 18):
        if result[i][j] > max:
            max = result[i][j]
            ind = j
    print(char[ind])


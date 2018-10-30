import scipy.misc
import glob
import numpy
import matplotlib.pyplot
import scipy.special

class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 가중치 행렬wih와 who (input_hidden 과 hidden_output)
        # 배열 내 가중치는 w_i_j로 표기. 노드 i에서 다음 계층의 노드 j로 연결됨을 의미.
        # w11 w21
        # w21 w22 등

        ##더 정교한 가중치 : 0을 중심으로 하고 1/sqrt(들어오는 연결노드의 개수) 의 표준편차를 가지는 정규분포에 따라 구한다.
        ##이를 위해서 numpy.random.normal() 함수를 쓰는데, (정규분포 중심, 표준편차, 넘파이 행렬) 순서로 들어가면 됨.
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        ##1/sqrt(~~) 가 pow(~~~) 임.
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        # 학습률
        self.lr = learningrate

        # 활성화 함수로 시그모이드 함수 이용
        self.activation_function = lambda x: scipy.special.expit(x)

    pass

    # 신경망 학습시키기
    def train(self, inputs_list, targets_list):
        # 입력리스트를 2차원 행렬로 변환
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 은닉 계층으로 들어오는 신호를 계산
        ##행렬곱은 dot으로 계산가능.
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 은닉 계층에서 나가는 신호를 계산 (시그모이드 함숫값)
        hidden_outputs = self.activation_function(hidden_inputs)

        # 최종 출력 계층으로 들어오는 신호륵 계산
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 최종 출력 계층에서 나가는 신호를 계산
        final_outputs = self.activation_function(final_inputs)

        # 오차는 (실제 값 - 계산 값)
        output_errors = targets - final_outputs
        # 은닉계층의 오차는 가중치에 의해 나뉜 출력 계층의 오차들을 재조합
        # T는 역으로 돌리는 개념
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # 은닉 계층과 출력 계층 간의 가중치 업데이트
        self.who = self.who + self.lr * numpy.dot((output_errors * final_outputs
                                                   * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        # 입력 계층과 은닉 계층 간의 가중치 업데이트
        self.wih = self.wih + self.lr * numpy.dot((hidden_errors * hidden_outputs
                                                   * (1.0 - hidden_outputs)), numpy.transpose(inputs))

    pass

    # 신경망에 질의하기
    def query(self, inputs_list):
        # 입력 리스트를 2차원 행렬로 변환
        inputs = numpy.array(inputs_list, ndmin=2).T
        # 은닉 계층으로 들어오는 신호를 계산
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 은닉 계층으로 나가는 신호를 계산
        hidden_outputs = self.activation_function(hidden_inputs)

        # 최종 출력 계층으로 들어오는 신호를 계산
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 최종 출력 계층에서 나가는 신호를 계산
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

#input노드, hidden노드, output노드의 개수설정
input_nodes = 784
hidden_nodes = 80
output_nodes = 4

#학습률 설정
learning_rate = 0.3

#신경망 생성
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

############################################################파일읽기###################################################################
own_train_data = []
record = []
#ㄱ
for i in range(70):
    image_file_name = 'fingerlanguage/1-train/'+str(i+1)+'.png'
   # print ("loading ... ", image_file_name)
    label = 1
    img_array = scipy.misc.imread(image_file_name, flatten = True)
    img_data  = 255.0 - img_array.reshape(784)
    img_data = (img_data / 255.0 * 0.99) + 0.01
    record = numpy.append(label, img_data)

    own_train_data.append(record)
    record = []
    pass
#ㄷ
record = []
for i in range(164):
    image_file_name = 'fingerlanguage/4-train/'+str(i+1)+'.png'
    #print ("loading ... ", image_file_name)
    label = 4
    img_array = scipy.misc.imread(image_file_name, flatten = True)
    img_data  = 255.0 - img_array.reshape(784)
    img_data = (img_data / 255.0 * 0.99) + 0.01
    record = numpy.append(label, img_data)

    own_train_data.append(record)
    record = []
    pass
#ㅍ
record = []
for i in range(176):
    image_file_name = 'fingerlanguage/13-train/'+str(i+1)+'.png'
    #print ("loading ... ", image_file_name)
    label = 13
    img_array = scipy.misc.imread(image_file_name, flatten = True)
    img_data  = 255.0 - img_array.reshape(784)
    img_data = (img_data / 255.0 * 0.99) + 0.01
    record = numpy.append(label, img_data)

    own_train_data.append(record)
    record = []
    pass
#ㅎ
for i in range(163):
    image_file_name = 'fingerlanguage/14-train/'+str(i+1)+'.png'
    #print ("loading ... ", image_file_name)
    label = 14
    img_array = scipy.misc.imread(image_file_name, flatten = True)
    img_data  = 255.0 - img_array.reshape(784)
    img_data = (img_data / 255.0 * 0.99) + 0.01
    record = numpy.append(label, img_data)
    #print(record)
    own_train_data.append(record)
    record = []
    pass

epochs = 10

for e in range(epochs):
    for item_train in range(len(own_train_data)):
        targets = own_train_data[item_train][0]
        #print(targets)
        inputs = own_train_data[item_train][1:]
        n.train(inputs, targets)
        pass

###################################테스트하기###############################33
own_test_data = []
record = []
#ㄱ
for i in range(10):
    image_file_name = 'fingerlanguage/1-train/'+str(i+1)+'.png'
    label = 1
    img_array = scipy.misc.imread(image_file_name, flatten = True)
    img_data  = 255.0 - img_array.reshape(784)
    img_data = (img_data / 255.0 * 0.99) + 0.01
    record = numpy.append(label, img_data)

    own_test_data.append(record)
    record = []
    pass
#ㄹ
for i in range(10):
    image_file_name = 'fingerlanguage/4-train/'+str(i+1)+'.png'
    label = 4
    img_array = scipy.misc.imread(image_file_name, flatten = True)
    img_data  = 255.0 - img_array.reshape(784)
    img_data = (img_data / 255.0 * 0.99) + 0.01
    record = numpy.append(label, img_data)

    own_test_data.append(record)
    record = []
    pass
#ㅍ
for i in range(10):
    image_file_name = 'fingerlanguage/13-train/'+str(i+1)+'.png'
    label = 13
    img_array = scipy.misc.imread(image_file_name, flatten = True)
    img_data  = 255.0 - img_array.reshape(784)
    img_data = (img_data / 255.0 * 0.99) + 0.01
    record = numpy.append(label, img_data)

    own_test_data.append(record)
    record = []
    pass
#ㅎ
for i in range(10):
    image_file_name = 'fingerlanguage/14-train/'+str(i+1)+'.png'
    label = 14
    img_array = scipy.misc.imread(image_file_name, flatten = True)
    img_data  = 255.0 - img_array.reshape(784)
    img_data = (img_data / 255.0 * 0.99) + 0.01
    record = numpy.append(label, img_data)

    own_test_data.append(record)
    record = []
    pass


for item_test in range(len(own_test_data)):
    correct_label = own_test_data[item_test][0]
    inputs = own_test_data[item_test][1:]

    outputs = n.query(inputs)
    print(outputs)
    label = numpy.argmax(outputs)
    print(label)


    pass
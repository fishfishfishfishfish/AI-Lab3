import numpy


def string_to_float(list_await=[]):
    res = []
    for k in range(len(list_await)):
        res.append(float(list_await[k]))
    return res


def cal_criterion(tp, tn, fp, fn, criterion_type):
    if tp == 0:
        print('tp = 0!!!!')
        return 0
    if criterion_type == 1:
        # Accuracy
        return (tp+tn)/(tp+tn+fp+fn)
    elif criterion_type == 2:
        # Precision
        return tp/(tp+fp)
    elif criterion_type == 3:
        # Recall
        return tp/(tp+fn)
    else:
        # F1
        p = tp/(tp+fp)
        r = tp/(tp+fn)
        return (2*p*r)/(p+r)


cri_name = ['Accuracy', 'Precision', 'Recall', 'F1']
ft = open('train.csv', 'r')
train_list_before = ft.readlines()
vc_len = len(train_list_before[0].split(','))  # 表示向量的维度大小
train_set_size = len(train_list_before)  # 训练集向量数量
train_x = []  # 训练集的增广特征向量的集合
train_y = []  # 训练集各个特征向量对应的结果
for line in train_list_before:
    temp_row = line.split(',')
    T = string_to_float(temp_row)
    T.insert(0, 1)  # 特征向量变成增广特征向量
    temp_np = numpy.array(T[0:vc_len])
    train_x.append(temp_np)
    train_y.append(T[vc_len])
w = numpy.ones(vc_len)  # 权重向量
w1 = numpy.ones(vc_len)  # 用于记录更新后的权重向量

print('更新几次停止？')
end_time = int(input())

# 更新end_time次结束
for i in range(end_time):
    for j in range(train_set_size):
        SUM = w.dot(train_x[j])
        w1 = w+train_x[j]*train_y[j]
        print('特征向量：', train_x[j])
        print('w:', w)
        print('正确结果：', train_y[j])
        if SUM*train_y[j] <= 0:
            w = w1.copy()
            print('预测错误，w更新为：', w)
            break
        else:
            print('预测正确')
    print('第', i, '次更新结束')
# 计算训练集的结果
TP = 0
TN = 0
FP = 0
FN = 0
for i in range(train_set_size):
    SUM = w.dot(train_x[i])
    if SUM > 0 and train_y[i] > 0:
        TP += 1
    elif SUM < 0 and train_y[i] < 0:
        TN += 1
    elif SUM > 0 and train_y[i] < 0:
        FP += 1
    else:
        FN += 1
cri_temp = numpy.ones(4)
for i in range(0, 4):
    cri_temp[i] = cal_criterion(TP, TN, FP, FN, i+1)
    print(cri_name[i], cri_temp[i])

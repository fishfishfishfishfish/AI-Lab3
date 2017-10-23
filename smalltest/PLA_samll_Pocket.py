import numpy


def string_to_float(list_await=[]):
    res = []
    for k in range(len(list_await)):
        res.append(float(list_await[k]))
    return res


def cal_criterion(tp, tn, fp, fn, criterion_type):
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


ft = open('train.csv', 'r')
train_list_before = ft.readlines()
vc_len = len(train_list_before[0].split(','))
train_x = []
train_y = []
for line in train_list_before:
    temp_row = line.split(',')
    T = string_to_float(temp_row)
    T.insert(0, 1)
    temp_np = numpy.array(T[0:vc_len])
    train_x.append(temp_np)
    train_y.append(T[vc_len])
w = numpy.ones(vc_len)
w1 = numpy.ones(vc_len)
set_size = len(train_x)
cri_type = 2  # 旋转判断是否更新的评价标准：accuracy，precision，recall， F1
# 初始化TP，TN，FP，FN
TP = 0
TN = 0
FP = 0
FN = 0
# 初始化最好的评价结果为初始的w得到的评价结果
for j in range(set_size):  # 遍历所有训练集的向量
    predict_res = w.dot(train_x[j])
    if predict_res > 0 and train_y[j] > 0:
        TP += 1
    elif predict_res < 0 and train_y[j] < 0:
        TN += 1
    elif predict_res >= 0 and train_y[j] < 0:
        FP += 1
    else:
        FN += 1
best_criterion = cal_criterion(TP, TN, FP, FN, cri_type)
print('w now:', w)
print('best_criterion now:', best_criterion)
flag = 0
while flag < set_size:  # flag标记下层循环是否遍历完所有训练向量
    flag = 0
    for i in range(set_size):  # 遍历所有训练向量
        pre_y = w.dot(train_x[i])*train_y[i]  # 对当前向量进行预测
        w1 = w + train_x[i]*train_y[i]  # 先行计算出更新后的w
        print('特征向量：', train_x[i])
        print('w:', w)
        print('预测结果', w.dot(train_x[i]))
        print('正确结果：', train_y[i])
        if pre_y <= 0:  # 预测错误
            print('预测错误')
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            for j in range(set_size):
                predict_res = w1.dot(train_x[j])
                print(' ', w1, ' · ', train_x[j], ' = ', predict_res, '---', train_y[j])
                if predict_res > 0 and train_y[j] > 0:
                    TP += 1
                elif predict_res < 0 and train_y[j] < 0:
                    TN += 1
                elif predict_res >= 0 and train_y[j] < 0:
                    FP += 1
                else:
                    FN += 1
            if TP != 0:  # TP为0时无法计算precision，recall，f1
                criterion = cal_criterion(TP, TN, FP, FN, cri_type)  # 计算w1的分类结果
                print('更新后效果：', criterion)
                print('当前最佳：', best_criterion)
                if criterion > best_criterion or pre_y == 0:  # w1的评价结果由于原来的
                    best_criterion = criterion   # 更新最好的结果
                    w = w1.copy()  # 深拷贝w
                    print('更新w：', w)
                    break
                else:
                    print('没有变好，不能更新')
        flag += 1  # 用于判断是否不能再更新了
    print(best_criterion)
print(w)
print(best_criterion)

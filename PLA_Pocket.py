import numpy


def string_to_float(list_await=[]):
    res = []
    for k in range(len(list_await)):
        res.append(float(list_await[k]))
    return res


def cal_criterion(tp, tn, fp, fn, criterion_type):
    if tp == 0 and criterion_type != 1:
        print('tp == 0!!!')
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
# debug:显示读取的训练集数据是否正确
# print(len(train_x))
# print(len(train_y))
# print(train_y)
ones_length = 30
wa = numpy.ones(ones_length)
wb = numpy.zeros(vc_len-ones_length)
wa_list = list(wa)
wb_list = list(wb)
w_list = wa_list + wb_list
w = numpy.array(w_list)
print('original w:', w)
w1 = numpy.ones(vc_len)
# dot向量点乘，（*）向量标乘
# print((w.dot(w1))*train_y[0])

# 不break的话，循环结束i=4000=len(train_x)
# i = 0
# for line in train_x:
#     i += 1
# print(i)
set_size = len(train_x)
cri_type = 4  # 旋转判断是否更新的评价标准：accuracy，precision，recall， F1
# 初始化TP，TN，FP，FN
TP = 0
TN = 0
FP = 0
FN = 0
# 初始化最好的评价结果为初始的w得到的评价结果
for j in range(set_size):  # 遍历所有训练集的向量
    predict_res = w.dot(train_x[j])
    if predict_res >= 0 and train_y[j] >= 0:
        TP += 1
    elif predict_res < 0 and train_y[j] < 0:
        TN += 1
    elif predict_res >= 0 and train_y[j] < 0:
        FP += 1
    else:
        FN += 1
best_criterion = cal_criterion(TP, TN, FP, FN, cri_type)
flag = 0
while flag < set_size:  # flag标记下层循环是否遍历完所有训练向量
    flag = 0
    for i in range(set_size):  # 遍历所有训练向量
        pre_y = w.dot(train_x[i])*train_y[i]  # 对当前向量进行预测
        w1 = w + train_x[i]*train_y[i]  # 先行计算出更新后的w
        if pre_y <= 0:  # 预测错误
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            for j in range(set_size):
                predict_res = w1.dot(train_x[j])
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
                if criterion > best_criterion or pre_y == 0:  # w1的评价结果由于原来的
                    best_criterion = criterion   # 更新最好的结果
                    w = w1.copy()  # 深拷贝w
                    break
        flag += 1  # 用于判断是否不能再更新了
    print(best_criterion)
print(w)
print(best_criterion)

# 获取验证集数据
fv = open('val.csv', 'r')
val_list_before = fv.readlines()
val_x = []
val_y = []
for line in val_list_before:
    temp_row = line.split(',')
    T = string_to_float(temp_row)
    T.insert(0, 1)  # 获取增广特征向量
    temp_np = numpy.array(T[0:vc_len])
    val_x.append(temp_np)
    val_y.append(T[vc_len])
# 进行验证
val_set_size = len(val_x)
VTP = 0
VTN = 0
VFP = 0
VFN = 0
for i in range(val_set_size):
    predict_res = w.dot(val_x[i])
    if predict_res > 0 and val_y[i] > 0:
        VTP += 1
    elif predict_res < 0 and val_y[i] < 0:
        VTN += 1
    elif predict_res >= 0 and val_y[i] < 0:
        VFP += 1
    else:
        VFN += 1
print('TP:', VTP, 'TN:', VTN, 'FP:', VFP, 'FN:', VFN)
accuracy = cal_criterion(VTP, VTN, VFP, VFN, 1)
precision = cal_criterion(VTP, VTN, VFP, VFN, 2)
recall = cal_criterion(VTP, VTN, VFP, VFN, 3)
F1 = cal_criterion(VTP, VTN, VFP, VFN, 4)
print('accuracy:', accuracy, 'precision:', precision, 'recall:', recall, 'F1:', F1)

# 测试集输出
ftest = open('test.csv', 'r')
fout = open('15352049_chenxinyu_PLA.csv', 'w')
test_list_before = ftest.readlines()
test_list = []
for line in test_list_before:
    T = []
    temp_row = line.split(',')
    for i in range(len(temp_row)-1):
        T.append(float(temp_row[i]))
    T.insert(0, 1.0)  # 塞1获取增广特征向量
    temp_np = numpy.array(T)
    test_list.append(temp_np)
for line in test_list:
    SUM = line.dot(w)  # 进行预测
    if SUM > 0:
        fout.write('1\n')
    else:
        fout.write('-1\n')

import numpy

def string_to_float(list_await=[]):
    res = []
    for k in range(len(list_await)):
        res.append(float(list_await[k]))
    return res

def cal_criterion(tp, tn, fp, fn, cri_type):
    if cri_type == 1:
        # Accuracy
        return (tp+tn)/(tp+tn+fp+fn)
    elif cri_type == 2:
        # Precision
        return tp/(tp+fp)
    elif cri_type == 3:
        # Recall
        return tp/(tp+fn)
    else:
        # F1
        P = tp/(tp+fp)
        R = tp/(tp+fn)
        return (2*P*R)/(P+R)


ft = open('train.csv', 'r')
train_list_before = ft.readlines()
vc_len = len(train_list_before[0].split(','))
train_x = []
train_y = []
for line in train_list_before:
    T = []
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
w = numpy.ones(vc_len)
w1 = numpy.ones(vc_len)
# dot向量点乘，（*）向量标乘
# print((w.dot(w1))*train_y[0])

# 不break的话，循环结束i=4000=len(train_x)
# i = 0
# for line in train_x:
#     i += 1
# print(i)
set_size = len(train_x)
best_criterion = 0
flag = 0
while flag <= vc_len:
    flag = 0
    for i in range(set_size):
        w1 = w + train_x[i]*train_y[i]
        if w.dot(train_x[i])*train_y[i] < 0:
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            for j in range(set_size):
                predict_res = w1.dot(train_x[j])
                if predict_res >= 0 and train_y[j] >= 0:
                    TP += 1
                elif predict_res < 0 and train_y[j] < 0:
                    TN += 1
                elif predict_res >= 0 and train_y[j] < 0:
                    FP += 1
                else:
                    FN += 1
            criterion = cal_criterion(TP, TN, FP, FN, 4)
            if criterion > best_criterion:
                best_criterion = criterion
                w = w1.copy()# 深拷贝
                break
        flag += 1
    print(best_criterion)
print(w)
print(best_criterion)

# 获取验证集数据
fv = open('val.csv', 'r')
val_list_before = fv.readlines()
val_x = []
val_y = []
for line in val_list_before:
    T = []
    temp_row = line.split(',')
    T = string_to_float(temp_row)
    T.insert(0, 1)
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
    if predict_res >= 0 and val_y[i] >= 0:
        VTP += 1
    elif predict_res < 0 and val_y[i] < 0:
        VTN += 1
    elif predict_res >= 0 and val_y[i] < 0:
        VFP += 1
    else:
        VFN += 1
print('TP:', VTP, 'TN:', VTN, 'FP:', VFP, 'FN:', FN)
accuracy = cal_criterion(VTP, VTN, VFP, VFN, 1)
precision = cal_criterion(VTP, VTN, VFP, VFN, 2)
recall = cal_criterion(VTP, VTN, VFP, VFN, 3)
F1 = cal_criterion(VTP, VTN, VFP, VFN, 4)
print('accuracy:', accuracy, 'precision:', precision, 'recall:', recall, 'F1:', F1)
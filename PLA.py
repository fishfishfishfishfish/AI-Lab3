def string_to_float(list_await=[]):
    res = []
    for k in range(len(list_await)):
        res.append(float(list_await[k]))
    return res


ft = open('train.csv', 'r')
vec_len = len(ft.readline().split(','))
w = []
for i in range(vec_len):
    w.append(1)
w1 = []
for i in range(vec_len):
    w1.append(0)
ft.close()

ft = open('train.csv', 'r')
train_list_before = ft.readlines()
train_list = []
for line in train_list_before:
    T = []
    temp_row = line.split(',')
    T = string_to_float(temp_row)
    train_list.append(T)
# print(train_list[0][2])
for line in train_list:
    line.insert(0, 1.0)
# print(train_list[200][2])
# print(len(train_list[200]))
# print(vec_len)

best_A = 0
best_P = 0
best_R = 0
best_F = 0
best_Aet = 0
best_Pet = 0
best_Ret = 0
best_Fet = 0
best_Aw = []
best_Pw = []
best_Rw = []
best_Fw = []
for end_time in range(100):
    w = []
    for i in range(vec_len):
        w.append(1)
    for i in range(end_time):
        for line in train_list:
            SUM = 0
            # print('line = ', line)
            # print('w = ', w)
            for j in range(vec_len):
                SUM += w[j]*line[j]
                w1[j] = w[j]+line[j]*line[vec_len]
            # print('SUM = ', SUM)
            # print('SUM*line[vec_len] = ', SUM*line[vec_len])
            if SUM*line[vec_len] < 0:
                # print('in')
                w = w1[:]
                break
        # print(w)
        # print(i)
    # 计算训练集的结果
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for line in train_list:
        SUM = 0
        for j in range(vec_len):
            SUM += w[j]*line[j]
        if SUM > 0 and line[vec_len] > 0:
            TP += 1
        elif SUM < 0 and line[vec_len] < 0:
            TN += 1
        elif SUM > 0 and line[vec_len] < 0:
            FP += 1
        else:
            FN += 1
    # print('end time = ', end_time);
    # print('TP:', TP, 'TN:', TN, 'FP:', FP, 'FN:', FN)
    Accuracy = (TP+TN)/(TP+TN+FP+FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = (2*Precision*Recall)/(Precision+Recall)
    # print('Accuracy:', Accuracy)
    # print('Precision:', Precision)
    # print('Recall:', Recall)
    # print('F1:', F1)
    if Accuracy > best_A:
        best_A = Accuracy
        best_Aet = end_time
        best_Aw = w[:]
    if Precision > best_P:
        best_P = Precision
        best_Pet = end_time
        best_Pw = w[:]
    if Recall > best_R:
        best_R = Recall
        best_Ret = end_time
        best_Rw = w[:]
    if F1 > best_F:
        best_F = F1
        best_Fet = end_time
        best_Fw = w[:]
print('best Accuracy:', best_A, 'in', best_Aet)
print('best Precision:', best_P, 'in', best_Pet)
print('best Recall:', Recall, 'in', best_Ret)
print('best F1:', F1, 'in', best_Fet)

# 进行验证
fv = open('val.csv', 'r')
val_list_before = fv.readlines()
val_list = []
for line in val_list_before:
    T = []
    temp_row = line.split(',')
    T = string_to_float(temp_row)
    val_list.append(T)
for line in val_list:
    line.insert(0, 1.0)
print('use best Accuracy')
VTP = 0
VTN = 0
VFP = 0
VFN = 0
for line in val_list:
    SUM = 0
    for j in range(vec_len):
        SUM += best_Aw[j]*line[j]
    if SUM > 0 and line[vec_len] > 0:
        VTP += 1
    elif SUM < 0 and line[vec_len] < 0:
        VTN += 1
    elif SUM > 0 and line[vec_len] < 0:
        VFP += 1
    else:
        VFN += 1
print('TP:', VTP, 'TN:', VTN, 'FP:', VFP, 'FN:', VFN)
# Accuracy = (VTP+VTN)/(VTP+VTN+VFP+VFN)
# Precision = VTP / (VTP + VFP)
# Recall = VTP / (VTP + VFN)
# F1 = (2*Precision*Recall)/(Precision+Recall)
# print('Accuracy:', Accuracy)
# print('Precision:', Precision)
# print('Recall:', Recall)
# print('F1', F1)

print('use best Precision')
VTP = 0
VTN = 0
VFP = 0
VFN = 0
for line in val_list:
    SUM = 0
    for j in range(vec_len):
        SUM += best_Pw[j]*line[j]
    if SUM > 0 and line[vec_len] > 0:
        VTP += 1
    elif SUM < 0 and line[vec_len] < 0:
        VTN += 1
    elif SUM > 0 and line[vec_len] < 0:
        VFP += 1
    else:
        VFN += 1
print('TP:', VTP, 'TN:', VTN, 'FP:', VFP, 'FN:', VFN)
# Accuracy = (VTP+VTN)/(VTP+VTN+VFP+VFN)
# Precision = VTP / (VTP + VFP)
# Recall = VTP / (VTP + VFN)
# F1 = (2*Precision*Recall)/(Precision+Recall)
# print('Accuracy:', Accuracy)
# print('Precision:', Precision)
# print('Recall:', Recall)
# print('F1', F1)

print('use best Recall')
VTP = 0
VTN = 0
VFP = 0
VFN = 0
for line in val_list:
    SUM = 0
    for j in range(vec_len):
        SUM += best_Rw[j]*line[j]
    if SUM > 0 and line[vec_len] > 0:
        VTP += 1
    elif SUM < 0 and line[vec_len] < 0:
        VTN += 1
    elif SUM > 0 and line[vec_len] < 0:
        VFP += 1
    else:
        VFN += 1
print('TP:', VTP, 'TN:', VTN, 'FP:', VFP, 'FN:', VFN)
# Accuracy = (VTP+VTN)/(VTP+VTN+VFP+VFN)
# Precision = VTP / (VTP + VFP)
# Recall = VTP / (VTP + VFN)
# F1 = (2*Precision*Recall)/(Precision+Recall)
# print('Accuracy:', Accuracy)
# print('Precision:', Precision)
# print('Recall:', Recall)
# print('F1', F1)

print('use best F1')
VTP = 0
VTN = 0
VFP = 0
VFN = 0
for line in val_list:
    SUM = 0
    for j in range(vec_len):
        SUM += best_Fw[j]*line[j]
    if SUM > 0 and line[vec_len] > 0:
        VTP += 1
    elif SUM < 0 and line[vec_len] < 0:
        VTN += 1
    elif SUM > 0 and line[vec_len] < 0:
        VFP += 1
    else:
        VFN += 1
print('TP:', VTP, 'TN:', VTN, 'FP:', VFP, 'FN:', VFN)
# Accuracy = (VTP+VTN)/(VTP+VTN+VFP+VFN)
# Precision = VTP / (VTP + VFP)
# Recall = VTP / (VTP + VFN)
# F1 = (2*Precision*Recall)/(Precision+Recall)
# print('Accuracy:', Accuracy)
# print('Precision:', Precision)
# print('Recall:', Recall)
# print('F1', F1)

ftest = open('test.csv', 'r')
fout = open('15352049_chenxinyu_PLA.csv','w')
test_list_before = ftest.readlines()
test_list = []
for line in test_list_before:
    T = []
    temp_row = line.split(',')
    for i in range(len(temp_row)-1):
        T.append(float(temp_row[i]))
    test_list.append(T)
for line in test_list:
    line.insert(0, 1.0)
for line in test_list:
    SUM = 0
    for j in range(vec_len):
        SUM += best_Fw[j]*line[j]
    if SUM >= 0:
        fout.write('1\n')
    else:
        fout.write('-1\n')
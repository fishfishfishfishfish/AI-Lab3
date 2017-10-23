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

best = numpy.zeros(4)  # 记录四个评测指标最优值：0-accuracy，1-precision，2-recall，3-F1
best_time = numpy.zeros(4)  # 记录四个评测指标取得最优值时采取的结束时间
best_w = []  # 记录取得最优评测指标时的w
for i in range(4):
    t_np = numpy.zeros(vc_len)
    best_w.append(t_np)
# 尝试多个结束时间
for end_time in range(100):
    w = numpy.ones(vc_len)  # 每次改变结束时间需要把w还原
    for i in range(end_time):  # 更新w向量end_time次
        for j in range(train_set_size):  # 遍历训练集的向量
            SUM = w.dot(train_x[j])  # 使用w的分类结果
            w1 = w+train_x[j]*train_y[j]  # 先行计算出w更新后的向量
            if SUM*train_y[j] <= 0:  # 如果之前的预测出错
                w = w1.copy()  # 更新w
                break  # 重头开始遍历训练集
    # 评价训练集的结果
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
    # print('end time = ', end_time);
    # print('TP:', TP, 'TN:', TN, 'FP:', FP, 'FN:', FN)
    cri_temp = numpy.ones(4)
    for i in range(0, 4):
        cri_temp[i] = cal_criterion(TP, TN, FP, FN, i+1)
        # print(cri_name[i], cri_temp[i])
        if cri_temp[i] > best[i]:
            best[i] = cri_temp[i]
            best_time[i] = end_time
            best_w[i] = w.copy()
for i in range(0, 4):
    print('best ', cri_name[i], ' ', best[i], ' in ', best_time[i], ' with ', best_w[i])
# 进行验证
# 获取验证集数据
fv = open('val.csv', 'r')
val_list_before = fv.readlines()
val_set_size = len(val_list_before)  # 验证集向量数量
val_x = []  # 验证集增广特征向量集合
val_y = []  # 验证集各个增广特征向量对应的结果
for line in val_list_before:
    temp_row = line.split(',')
    T = string_to_float(temp_row)
    T.insert(0, 1)  # 特征向量前面塞1变成增广特征向量
    temp_np = numpy.array(T[0:vc_len])
    val_x.append(temp_np)
    val_y.append(T[vc_len])
for cri_it in range(4):
    print('use best ', cri_name[cri_it])
    VTP = 0
    VTN = 0
    VFP = 0
    VFN = 0
    for i in range(val_set_size):
        SUM = best_w[cri_it].dot(train_x[i])
        if SUM > 0 and train_y[i] > 0:
            VTP += 1
        elif SUM < 0 and train_y[i] < 0:
            VTN += 1
        elif SUM > 0 and train_y[i] < 0:
            VFP += 1
        else:
            VFN += 1
    print('TP:', VTP, 'TN:', VTN, 'FP:', VFP, 'FN:', VFN)
    cri_temp = numpy.zeros(4)
    for val_cri_it in range(4):
        cri_temp[val_cri_it] = cal_criterion(VTP, VTN, VFP, VFN, val_cri_it+1)
        print(cri_name[val_cri_it], cri_temp[val_cri_it])
# 对测试集进行分类
ftest = open('test.csv', 'r')
fout = open('15352049_chenxinyu_PLA.csv', 'w')
test_list_before = ftest.readlines()
test_list = []  # 测试集增广特征向量的集合
for line in test_list_before:
    T = []
    temp_row = line.split(',')  # 计算一行向量的维数
    # 减一是为了防止读到问号
    for i in range(len(temp_row)-1):
        T.append(float(temp_row[i]))
    T.insert(0, 1.0)  # 塞1变成增广特征向量
    temp_np = numpy.array(T)
    test_list.append(temp_np)
for line in test_list:
    SUM = line.dot(best_w[3])  # 采用出现最优F1值的w权重
    if SUM > 0:
        fout.write('1\n')
    else:
        fout.write('-1\n')

1. 初始化增广特征向量
for x in '训练集的向量' :
    x.append(0,1)
w = [1, 1, 1, ..'与x维度相同'.., 1, 1, 1]

2. 迭代n次更新w
for i = 0:n
    for x in '训练集的向量'
        predict_result = x · w
        y = 'x对应的正确结果'
        if predict_result * y <= 0:
            w = w + x * y
            
3. 用于验证集
TP = 0
TN = 0
FP = 0
FN = 0
for x in '验证集的向量':
    predict_result = x · w
    y = 'x对应的正确结果'
    if predict_result > 0 and y > 0:
        TP++
    else if predict_result < 0 and y < 0:
        TF++
    else if predict_result > 0 and y > 0:
        FP++
    else:
        FN++
'计算评价指标'(TP, TN, FP, FN)

4. 评价指标的计算
if tp == 0:
    return 0
Accuracy = (tp+tn)/(tp+tn+fp+fn)
Precision = tp/(tp+fp)
Recall = tp/(tp+fn)
F1 = (2*Precision*Recall)/(Precision+Recall)

5. 口袋算法

for x in '训练集的向量':
    predict_result = x · w
    y = 'x对应的正确结果'
    '累加TP, TN, FP, FN'
best_criterion = '计算评价指标'(TP, TN, FP, FN)
while '下层循环遍历了所有向量x'
    for x in '训练集的向量':
        predict_result = x · w
        y = 'x对应的正确结果'
        if predict_result * y < 0
            w1 = w + x * y
            for xi in '训练集的向量'
                predict_result = w1 · xi
                y = 'xi对应的正确结果'
                '累加TP, TN, FP, FN'
            criterion = '计算评价指标'(TP, TN, FP, FN)
            if criterion > best_criterion:
                best_criterion = criterion
                w = w1
                break

6. 用于测试集
for x in '测试集的向量':
    x.append(0, 1)
for x in '测试集的向量':
    predict_result = x · w
    if predict_result > 0:
        output << 1
    else:
        output << -1
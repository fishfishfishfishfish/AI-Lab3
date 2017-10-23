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
        TP ++
    
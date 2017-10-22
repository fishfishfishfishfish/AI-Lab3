f = open("train.csv")
vec_len = len(f.readline().split(','))
w = []
for i in range(vec_len):
    w.append(1)
print(vec_len, len(w))

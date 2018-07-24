
import csv
import random
#直接读取打好的标签的数据
csv_read = csv.reader(open("C:/Users/橘子/Desktop/ZC/date/trainshuju/train_wx.csv"))
csv_read_y = csv.reader(open("C:/Users/橘子/Desktop/ZC/date/trainshuju/train_wy.csv"))

train_data = []
for f_line in csv_read:
    l_d = []
    for l in f_line:
        l_d.append(float(l))
    train_data.append(l_d)
print('训练数据train_data长度',len(train_data))
Y_date = []
for line in csv_read_y:
    l_d = []
    for l in line:
        l_d.append(int(l))
    Y_date.append(l_d)
print('训练数据Y_data长度',len(Y_date))

# 随机打乱
cc = list(zip(Y_date, train_data))

random.shuffle(cc)

Y_date[:], train_data[:] = zip(*cc)



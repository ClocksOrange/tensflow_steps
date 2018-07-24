import csv
import numpy as np
index_cut = [[[1],[350,550], [650,850], [830,1030], [1000,1200]],
    [[2],[350,550], [680,880], [900,1100], [1140,1340]],
    [[3],[350,550], [650,850], [900,1100], [1250,1450]],
    [[4],[350,550], [700,900], [900,1100], [1400,1600]],
    [[5],[250,450], [550,750], [900,1100], [1250,1450]],
    [[6],[250,450], [560,760], [900,1100], [1340,1540]],
    [[7],[350,550], [570,770], [900,1100], [1130,1330]],
    [[8],[250,450], [570,770], [900,1100], [1280,1480]],
    [[9],[250,450], [550,750], [900,1100], [1100,1300]],
    [[10],[250,450], [550,750], [900,1100], [1300,1500]],
    [[11],[250,450], [550,750], [900,1100], [1050,1250]],
    [[12],[350,550], [650,850], [900,1100], [1150,1350]],
    [[13],[350,550], [600,800], [900,1100], [1450,1650]],
    [[14],[350,550], [600,800], [900,1100], [1300,1500]],
    [[15],[350,550], [630,830], [900,1100], [1250,1450]],
    [[16],[350,550], [600,800], [900,1100], [1450,1650]],
    [[17],[350,550], [550,750], [900,1100], [1230,1430]],
    [[18],[350,550], [650,850], [900,1100], [1300,1500]],
    [[19],[350,550], [650,850], [900,1100], [1150,1350]],
    [[20],[350,550], [550,750], [900,1100], [1420,1620]],
    [[21],[350,550], [650,850], [900,1100], [1450,1650]],
    [[22],[350,550], [550,750], [900,1100], [1600,1800]],
    [[23],[350,550], [700,900], [900,1100], [1260,1460]],
    [[24],[350,550], [750,950], [1150,1350], [1600,1800]],
    [[25],[350,550], [700,900], [900,1100], [1500,1700]],
    [[26],[350,550], [650,850], [900,1100], [1250,1450]],
    [[27],[350,550], [700,900], [900,1100], [1380,1580]],
    [[28],[350,550], [650,850], [900,1100], [1150,1350]],
    [[29],[350,550], [800,1000], [1100,1300], [1430,1630]],
    [[30],[350,550], [650,850], [900,1100], [1250,1450]],
    [[31],[350,550], [700,900], [900,1100], [1780,1980]],
    [[32],[350,550], [700,900], [900,1100], [1140,1340]],
    [[33],[350,550], [650,850], [900,1100], [1200,1400]],
    [[34],[350,550], [650,850], [900,1100], [1250,1450]],
    [[35],[350,550], [740,940], [1000,1200], [1400,1600]],
    [[36],[350,550], [680,880], [900,1100], [1550,1750]],
    [[37],[350,550], [750,950], [1000,1200], [1540,1740]],
    [[38],[350,550], [800,1000], [1200,1400], [1820,2020]],
    [[39],[350,550], [780,980], [1000,1200], [1350,1550]],
    [[40],[350,550], [750,950], [900,1100], [1350,1550]],
    [[41],[350,550], [750,950], [1000,1200], [1450,1650]],
    [[42],[350,550], [770,970], [1000,1200], [1550,1750]],
    [[43],[350,550], [700,900], [900,1100], [1500,1700]],
    [[44],[350,550], [740,940], [1000,1200], [1640,1840]],
    [[45],[350,550], [730,930], [1000,1200], [1450,1650]],
    [[46],[350,550], [700,900], [900,1100], [1450,1650]],
    [[47],[350,550], [650,850], [900,1100], [1450,1650]],
    [[48],[350,550], [700,900], [900,1100], [1500,1700]],
    [[49],[350,550], [650,850], [900,1100], [1400,1600]],
    [[50],[350,550], [660,860], [900,1100], [1230,1430]],
]

zc = [[[1],[250,450], [550,750], [750,950], [950,1150]],
[[2],[250,450], [500,700], [700,900], [900,1100]],
[[3],[250,450], [530,730], [720,920], [930,1130]],
[[4],[1600,1800], [1950,2150], [2200,2400], [2550,2750]],
[[5],[200,400], [450,650], [720,920], [1100,1300]],
[[6],[200,400], [450,650], [720,920], [1100,1300]],
[[7],[200,400], [450,650], [720,920], [1100,1300]],
[[8],[200,400], [450,650], [720,920], [1150,1350]],
[[9],[200,400], [480,680], [720,920], [1000,1200]],
[[10],[200,400], [480,680], [720,920], [1100,1300]],
[[11],[200,400], [430,630], [720,920], [1100,1300]],
[[12],[200,400], [430,630], [720,920], [930,1130]],
[[13],[200,400], [430,630], [720,920], [1000,1200]],
[[14],[200,400], [430,630], [720,920], [1100,1300]],
[[15],[200,400], [450,650], [720,920], [950,1150]],
[[16],[200,400], [400,600], [720,920], [930,1130]],
[[17],[200,400], [450,650], [720,920], [950,1150]],
[[18],[200,400], [500,700], [720,920], [930,1130]],
[[19],[200,400], [490,690], [720,920], [1050,1250]],
[[20],[200,400], [450,650], [720,920], [1050,1250]],
[[21],[200,400], [500,700], [720,920], [1050,1250]],
[[22],[200,400], [450,650], [720,920], [950,1150]],
[[23],[200,400], [450,650], [720,920], [1050,1250]],
[[24],[200,400], [450,650], [720,920], [950,1150]],
[[25],[200,400], [580,780], [720,920], [1130,1330]],
[[26],[200,400], [500,700], [720,920], [1250,1450]],
[[27],[200,400], [500,700], [720,920], [1050,1250]],
[[28],[200,400], [450,650], [720,920], [950,1150]],
[[29],[200,400], [500,700], [720,920], [1050,1250]],
[[30],[200,400], [490,690], [720,920],  [950,1150]],
[[31],[200,400], [440,640], [720,920], [950,1150]],
[[32],[200,400], [550,750], [720,920], [1100,1300]],
[[33],[350,550], [680,880], [900,1100], [1340,1540]],
[[34],[200,400], [450,650], [720,920], [950,1150]],
[[35],[200,400], [520,720], [720,920], [1040,1240]],
    ]

wak = [[[2],[200,400], [450,650], [600,700], [750,950]],
[[3],[200,400], [500,700], [650,750], [800,1000]],
[[5],[200,400], [500,700], [650,750], [800,1000]],
[[6],[200,400], [500,700], [650,750], [750,950]],
[[7],[200,400], [500,700], [650,750], [750,950]],
[[8],[200,400], [500,700], [650,750], [750,950]],
[[9],[200,400], [500,700], [650,750], [750,950]],
[[10],[200,400], [500,700], [650,750], [750,950]],
[[11],[200,400], [500,700], [650,750], [750,950]],
[[12],[200,400], [500,700], [650,750], [750,950]],
[[13],[200,400], [500,700], [650,750], [850,1050]],
[[14],[200,400], [500,700], [650,750], [850,1050]],
[[15],[200,400], [500,700], [650,750], [800,1000]],
[[16],[200,400], [500,700], [650,750], [800,1000]],
[[17],[200,400], [500,700], [650,750], [850,1050]],
[[18],[200,400], [600,800], [650,750], [950,1150]],
[[19],[200,400], [600,800], [650,750], [900,1100]],
[[20],[200,400], [550,750], [650,750], [900,1100]],
[[21],[200,400], [530,730], [650,750], [950,1150]],
[[22],[200,400], [500,700], [650,750], [850,1050]],
[[23],[200,400], [600,800], [650,750], [900,1100]],
[[24],[200,400], [600,800], [650,750], [950,1150]],
[[25],[200,400], [600,800], [650,750], [950,1150]],
[[26],[200,400], [550,750], [650,750], [900,1100]],
[[27],[200,400], [600,800], [650,750], [950,1150]],
[[28],[200,400], [550,750], [650,750], [950,1150]],
[[29],[200,400], [600,800], [650,750], [950,1150]],
[[30],[200,400], [550,750], [650,750], [900,1100]],
[[31],[200,400], [600,800], [650,750], [1100,1300]],
[[32],[200,400], [600,800], [650,750], [950,1150]],
[[33],[200,400], [600,800], [650,750], [950,1150]],
[[34],[200,400], [580,780], [650,750], [1000,1200]],
[[35],[200,400], [620,820], [650,750], [1050,1250]],
[[36],[200,400], [600,800], [650,750], [1000,1200]],
[[37],[200,400], [600,800], [650,750], [1150,1350]],
[[38],[200,400], [600,800], [650,750], [1000,1200]],
[[39],[200,400], [600,800], [650,750], [1050,1250]],
[[40],[200,400], [600,800], [650,750], [1000,1200]],
[[41],[200,400], [600,800], [650,750], [1000,1200]],
[[42],[200,400], [600,800], [650,750], [1000,1200]],
[[43],[200,400], [600,800], [650,750], [1100,1300]],
[[44],[200,400], [600,800], [650,750], [1050,1250]],
[[45],[200,400], [550,750], [650,750], [1000,1200]],
[[46],[200,400], [500,700], [650,750], [950,1150]],
[[47],[200,400], [600,800], [650,750], [1050,1250]],
[[48],[200,400], [600,800], [650,750], [1100,1300]],
[[49],[200,400], [600,800], [650,750], [1100,1300]],
[[50],[200,400], [600,800], [650,750], [1000,1200]],

       ]

#print(index_cut)
#直接按照标签读取原始数据，形成训练数据
def Direct_read():
    train_data = []
    Y_date = []
    for line_cut in index_cut:
        # print(line_cut[0][0])
        csv_read = csv.reader(open("C:/Users/橘子/Desktop/ZC/date/trainshuju/左侧走_%d.csv" % (line_cut[0][0])))
        data = []
        for f_line in csv_read:
            data.append(f_line)

        k = 0
        for j in line_cut[1:5]:
            # print(j)
            data_line = []

            #        if k==2:       #当wak,执行句
            #            k += 1
            #            continue
            for line in data[j[0] - 1:j[1]]:
                #  print(line[9],line[14])
                data_line.append(float(line[9]))  # csv读取的为字符
            # print(data_line)
            # data_line为一个输入样例
            train_data.append(data_line)
            # 为每个样例处理标签
            Y_line = [0, 0, 0, 0]
            Y_line[k] = 1
            #   print(k,Y_line)
            k += 1
            Y_date.append(Y_line)

    csv_File_x = open("C:/Users/橘子/Desktop/ZC/date/trainshuju/train_x.csv ", "a+", newline='')
    writer_x = csv.writer(csv_File_x)
    csv_File_y = open("C:/Users/橘子/Desktop/ZC/date/trainshuju/train_y.csv ", "a+", newline='')
    writer_y = csv.writer(csv_File_y)

    len_train = len(train_data)
    print(len_train, train_data)
    len_y = len(Y_date)
    print(len_y, Y_date)

    for k in range(len_train):
        writer_x.writerow(train_data[k])
        writer_y.writerow(Y_date[k])

lab =index_cut
def restart_lab():
    lab_= []
    for line in lab:
        line[2][0]+=10
        line[2][1]-=30
        line[4][0]+=30
        line[4][1]-=10
        line[1][1]=line[2][0]
        line[3][0]=line[2][1]
        line[3][1]=line[4][0]
        lab_.append(line)
    list.copy(lab_)

#通过滑动窗口打成不同的标签
def sli_window():
    restart_lab()
    l_slip = 240
    move = 20
    train_data = []
    Y_date = []

    def chijiuhua():
        # 持久化
        csv_File_x = open("C:/Users/橘子/Desktop/ZC/date/trainshuju/train_wx.csv ", "a+", newline='')
        writer_x = csv.writer(csv_File_x)
        csv_File_y = open("C:/Users/橘子/Desktop/ZC/date/trainshuju/train_wy.csv ", "a+", newline='')
        writer_y = csv.writer(csv_File_y)
        print(len(train_data),len(Y_date))

        for k in range(len(Y_date)):
            writer_x.writerow(train_data[k])
            writer_y.writerow(Y_date[k])

    for line_cut in lab:
        csv_read = csv.reader(open("C:/Users/橘子/Desktop/ZC/date/trainshuju/左侧走_%d.csv" % (line_cut[0][0])))
        data = [] # 保存一个文件的数据
        for f_line in csv_read:
            data.append(f_line)
        #增加一个标签,目前是一个段连续的数据
        line_cut.append([line_cut[4][1],line_cut[4][1]+200])
        #print(line_cut)

        for L in range(line_cut[1][0],line_cut[4][1]+200-l_slip,move): #l_slip为窗口大小,move为窗口移动大小
            R = L + l_slip  # 窗口右边
            data_line = []  # 一个待训练的数据
            for line in data[L:R]:
                data_line.append(float(line[9]))
            train_data.append(data_line)

            # 为每个data_line样例处理标签
            Y_line = [0, 0, 0, 0] #一个标签
            for j in range(1,6,1):  #j:1,2,3,4,5   L R为当前滑动窗口
                t=j-1
                if t==4:
                    t=0
                '标签在窗口里'
                if L <= line_cut[j][0] and line_cut[j][1]<= R:
                 #   print('标签在窗口里-wL,wR,ll,lr->', L, R, line_cut[j][0], line_cut[j][1])
                    Y_line[t] = 1
                    break
                    '窗口左边处于上一个标签，右边处于下一个标签'
                elif L < line_cut[j][0] and line_cut[j][0] < R:
                  #  print('wL,wR,ll,lr->', L, R, line_cut[j][0], line_cut[j][1])
                    if line_cut[j][0] - L >= R - line_cut[j][0] and j - 2 >= 0:
                        Y_line[t-1] = 1
                    else:
                        Y_line[t] = 1
                    break

                    '窗口在标签里'
                elif line_cut[j][0]<= L and R <= line_cut[j][1]:
                    Y_line[t] =1
                 #   print('窗口在标签2-wL,wR,ll,lr->', L, R, line_cut[j][0], line_cut[j][1])
                    break
                else:
                    #if L <= line_cut[j][1] and line_cut[j][1] <= R:
                    #        print('2-wL,wR,ll,lr->', L, R, line_cut[j][0], line_cut[j][1])
                    #else:
                     #   print('orther-wL,wR,ll,lr->',L,R,line_cut[j][0],line_cut[j][1])
                    continue
            #   print(Y_line)
            Y_date.append(Y_line)

    #chijiuhua()

#sli_window()

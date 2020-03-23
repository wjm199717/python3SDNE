import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm



def get_data(file):
    fin = open(file)
    for i in range(3):  
        fin.readline()
    x = []
    y = []
    for line in fin.readlines():
        line = line.strip().split(' ')
        if len(line) < 3:
            break
        x.append(float(line[2]))
        line = line[3].split('[')[1].split(',')[0]
        y.append(float(line))
    return x, y

if __name__ == "__main__":    
    #os.getcwd()返回当前进程的工作目录。
    #os.sep()返回的是分割符'\\'
    cwd = os.getcwd() + os.sep + 'Log'
#rt是遍历文件的根目录，dirs是一个列表存储的是根目录下的子目录（目录也就是文件夹），
#files也是一个列表存储的是根目录下的文件
#比如下面第一次遍历时根目录是这个log目录，第二次遍历时根目录就是dies列表中的一个目录，然后依次遍历下去
#os.walk返回值是一个生成器(generator),也就是需要不断的遍历它，来获得所有的内容。
    for rt, dirs, files in os.walk(cwd):
        #np.linspace用来创建等差数列
        cmap = cm.rainbow(np.linspace(0, 1, len(files)))
        for i in range(len(files)):
            x, y = get_data(cwd + os.sep + files[i])
            #plt.plot绘制折线图，“label”指定线条的标签
            #用plt.plot就默认有一个画布了,也就是plt.figure
            #plt.figure就是默认生成一个画布，plt.show就是将这个画布显示出来（包括画布上的内容）
            plt.plot(x, y, color = cmap[i], label = files[i][0:-8])
    #plt.legend()就是给画的图加图例
    plt.legend()
    plt.ylabel("cost")
    plt.xlabel("time(s)")
    #plt.show()
    plt.savefig("picture" + os.sep + "TimeCompare.png")
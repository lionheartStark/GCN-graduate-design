
import pickle
import matplotlib.pyplot as plt
import numpy as np

def get_pic():
    epoches = 1000
    ep = [i for i in range(50, epoches+1, 50)]
    print(ep)

    a=[0.8,0.2]
    b=a
    c=a
    d=a
    g = [a, b, c, d]
    #pickle.dump(g, open('res_' + f'{epoches}', 'wb'))
    with open('res_' + f'{epoches}', "rb") as f:
        g = pickle.load(f)
    a, b, c, d = g

    ep = [i for i in range(50, epoches + 1, 50)]
    plt.plot(np.array(ep), np.array(a), 'r', label='Train loss')
    plt.plot(np.array(ep), np.array(b), 'g', label='Valid loss')
    plt.plot(np.array(ep), np.array(c), 'black', label='Illicit F1')
    plt.plot(np.array(ep), np.array(d), 'orange', label='F1')
    plt.legend(['Train loss', 'Valid loss', 'Illicit F1', 'F1'])
    plt.ylim([0, 1.0])
    plt.xlim([50, 1001])
    plt.savefig("mk_"+"filename.png")
    plt.show()
if __name__ == '__main__':
    get_pic()

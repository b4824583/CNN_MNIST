from mnist import MNIST
import math
import numpy as np
def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig

import random
import matplotlib.pyplot as plt




from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

# plt.imshow(test_img)
# plt.show()



from numba import jit
lr=0.00001
@jit
def tanh_differential(x):
    tanh_diff=1-(np.tanh(x)*np.tanh(x))

    return tanh_diff
def backward2(img,filter1,feat_map1,w2,delta2):

    """
    look each feature map1 unit
    
    """
    delta1=np.zeros((feat_map1.shape[0],feat_map1.shape[1],feat_map1.shape[2]))
    count=0
    for i in range(feat_map1.shape[0]):
        for j in range(feat_map1.shape[1]):
            for k in range(feat_map1.shape[2]):
                ##################
                """
                    each feature map run the filter1
                """
                # sum=0
                # delta1=0

                """
                sum the each delta2 * w2
                
                """
                for num in range(w2.shape[0]):
                    # w2[num][count]
                    dif_feat_unit=tanh_differential(feat_map1[i][j][k])
                    delta1[i][j][k]=delta1[i][j][k]+delta2[num]*w2[num][count]*dif_feat_unit
                count+=1
                for l in range(filter1.shape[1]):
                    for m in range(filter1.shape[2]):
                        filter1[i][l][m]=filter1[i][l][m]-delta1[i][j][k]*img[j+l][k+m]*lr
    return filter1,delta1

def backward1(x1,w2,pred,ans):
    delta=np.zeros(pred.shape[0])

    for i in range(ans.shape[0]):
        delta[i] = (-2) * (ans[i] - pred[i])
        for j in range(w2.shape[1]):
            w2[i][j]=w2[i][j]-(delta[i]*x1[j]*lr)


    # delta=(-2)*(ans[5]-pred[5])
    # for j in range(w2.shape[1]):
    #     w2[5][j]=w2[5][j]-(delta*x1[j]*lr)

    return w2,delta

def forward(img,filter1,feat_maps1,x1,w2):

    for n in range(maps_num):
        for row in range(24):
            for col in range(24):
                sum = 0
                for i in range(cov_ker_row):
                    for j in range(cov_ker_col):
                        # print(str(i)+" "+str(col)+"-----"+str(j)+" "+str(row))

                        sum = sum + img[i + col][j + row] * filter1[n][i][j]
                feat_maps1[n][row][col] =sum
        # print("###\t\t" + str(n) + "\t\t###")
        # print("-------------------------")
    count = 0
    """
        s1->x1 process
    """
    for n in range(feat_maps1.shape[0]):
        for i in range(feat_maps1.shape[1]):
            for j in range(feat_maps1.shape[2]):
                x1[count] = np.tanh(feat_maps1[n][i][j])
                count += 1
    # print(x1.shape)
    pred = np.zeros(10)
    for i in range(w2.shape[0]):
        pred[i] = w2[i].T.dot(x1)
    # print(pred)

    return x1,w2,pred


if __name__=='__main__':
    print ("Executed when invoked directly")
    # import cupy as cp
    # x_on_gpu0 = cp.array([1, 2, 3, 4, 5])
    # exit()
    mndata =MNIST("samples/")
    np.random.seed(0)
    images,label=mndata.load_training()
    width=28
    length=28

    TrainNum=10

    img=np.zeros((TrainNum,width,length))
    maps_num=6
    maps_col=24
    maps_row=24
    feat_maps1=np.zeros((maps_num,maps_row,maps_col))

    # padding_zeros=np.zeeros((30,30))
    # print(size(images))
    #60000 * 784
    # print((images[0]))
    ans = np.zeros((img.shape[0],10))
    for i in range(img.shape[0]):
        count=0
        for j in range(width):
            for k in range(length):
                img[i][j][k]=images[i][count]/255
                ans[i][label[i]]=1.0
                # plt.imshow(img[i])
                # plt.show()
                # exit()
                count+=1

    #fliter
    # plt.imshow(img[15])
    # plt.show()
    # print(ans[15])
    # exit()
    cov_ker_col=5
    cov_ker_row=5
    cov_ker=25
    """
     filter 1 = w1
    """
    filter1=np.random.randn(maps_num,cov_ker_row,cov_ker_col)
    # print(filter[0])
    """
     I need design a filter moveing way
      filter 1 
      
    """
    # plt.imshow(img)
    # plt.show()
    x1_num = feat_maps1.shape[0] * feat_maps1.shape[1] * feat_maps1.shape[2] + 1
    w2 = np.random.randn(10, x1_num)
    x1 = np.ones(x1_num)



    # exit()
    # hori_stride=1
    # vert_stride=1
    for epoch in range(3500):
        print("epoch:"+str(epoch))
        for itr in range(img.shape[0]):
            # continue



            x1,w2,pred=forward(img[itr],filter1,feat_maps1,x1,w2)
            print("\titer:"+str(itr)+"\tloss:\t"+str(np.sum(pred)))
            w2,delta2=backward1(x1,w2,pred,ans[itr])
            '''
            x0=img
            w1=filter
            s1=featture map
            '''
            filter1,delta1=backward2(img[itr],filter1,feat_maps1,w2,delta2)
    with open('weight.npy', 'wb') as f:
        np.save(f,w2)
        np.save(f,filter1)
    for i in range(5):
        test_id=random.randint(0, 10)
        plt.imshow(img[test_id])
        plt.show()
        x1,w2,pred=forward(img[test_id],filter1,feat_maps1,x1,w2)
        print(test_id)
        print("0:"+str(pred[0]))
        print("1:"+str(pred[1]))
        print("2:"+str(pred[2]))
        print("3:"+str(pred[3]))
        print("4:"+str(pred[4]))
        print("5:"+str(pred[5]))
        print("6:"+str(pred[6]))
        print("7:"+str(pred[7]))
        print("8:"+str(pred[8]))
        print("9:"+str(pred[9]))

else:
    print ("Executed when import ")

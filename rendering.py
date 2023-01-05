import numpy as np
import cv2
import msvcrt
import math
from math import pi, sin, cos
from matplotlib import pyplot as plt   #显示图像等用，若需要用几个演示函数则启用
from mpl_toolkits.mplot3d import Axes3D  #同上

def rendering(dir):
    # z的尺度与x和y相同，大小等同于测试图像大小，位置与测试图像像素点一一对应
    # imgs为渲染结果，大小等同于测试图像大小，位置与测试图像像素点一一对应
    z = np.zeros([168, 168])

    img_S = ReadS(dir)  # img_S存储光源矩阵[3,7]，其每一列为光源在坐标系下单位向量
    imgx, albedo = ReadX(dir)  # img_x[n,7]为原始图片，average为7张图片所有照度平均值
    # img_valid为图片像素有效值矩阵[n,7] 值为1代表对应像素有效，0无效
    img_valid = CheckValid(imgx, albedo)
    # img_bg[n]为图片背景判定矩阵，1代表对应像素不是背景，（目前算法为有三个有效值即非背景），img-valicount[n]存储对应像素有几个有效值
    img_bg, img_validcount = CheckBG(img_valid)
    img_b = Step1(imgx, img_S, img_valid, img_bg)  # img_b[n,3]即对应B

    imgs,testS = render(dir, img_b, img_bg)
    
    z=inter(img_b)
    
    z=OptimizeZ(z)  #对z进行对称优化，尺度修正

    show3d(z) #显示z的三维模型用
    sh = shadow(z, testS ,img_bg) #阴影优化
    ShowShadowMap(sh)
    # ShowBGMap(img_bg)    #去注释可启用查看目前人脸哪些像素被认定为背景
    # ShowValidMap(img_valid)  #去注释可启用查看目前人脸各图有效值情况
    # msvcrt.getch()
    return z, imgs

def ReadS(dir):  # 读取光源的函数 返回[3,7]的光源
    tempS = np.zeros([3, 7]).astype(np.float)
    with open(dir+'/train.txt', 'r') as file:
        for i in range(0, 7):
            lines = file.readline()
            _, a, b = (float(x) for x in lines.split(','))
            tempS[0][i] = math.sin(a*3.1416/180)*(math.cos(b*3.1416/180))
            tempS[1][i] = math.sin(b*3.1416/180)
            tempS[2][i] = math.cos(b*3.1416/180)*(math.cos(a*3.1416/180))
    return tempS


def ReadX(dir):  # 读取图片
    train_img_read = np.zeros([168, 168, 7]).astype(np.float)
    imgx = np.zeros([168*168, 7]).astype(np.float)
    albedo=np.zeros(168*168).astype(np.float)
    for i in range(0, 7):
        train_img_read[:, :, i] = cv2.imread(
            dir+'/train/'+str(i+1)+'.bmp',cv2.IMREAD_GRAYSCALE)  # 读取图片，这里读进来是三通道
        imgx[:, i] = train_img_read[:, :,  i].T.flatten()  # 二维数组展开为一维，对应文献公式中大x
    for i in range(0,168*168):
        albedo[i]=np.mean(imgx[i])
    return imgx, albedo


def CheckValid(imgx, albedo):
    img_valid = np.zeros([168*168, 7]).astype(np.bool)
    for i in range(0, 168*168):
        for j in range(0, 7):
            if (imgx[i, j]/albedo[i]<0.1) or imgx[i,j]>253:
                img_valid[i, j] = 0
            else:
                img_valid[i, j] = 1
    return img_valid


def ShowValidMap(img_valid):
    img_validmap = img_valid.reshape((168, 168, 7))
    for i in range(0, 7):
        plt.figure("Image")
        plt.imshow(img_validmap[:, :, 1], cmap='gray')
        plt.axis('on')
        plt.title('image')
        plt.show()
    msvcrt.getch()

def ShowShadowMap(sh):
    # sh = sh.reshape((10,168, 168))
    for i in range(0, 10):
        plt.figure("Image")
        plt.imshow(sh[i,:,:], cmap='gray')
        plt.axis('on')
        plt.title('image')
        plt.show()
    msvcrt.getch()

def CheckBG(img_valid):
    img_bg = np.zeros([168*168]).astype(np.uint8)
    img_validcount = np.zeros([168*168]).astype(np.uint8)
    for i in range(0, 168*168):
        count = 0
        for j in range(0, 7):
            if img_valid[i, j] > 0:
                count += 1
        if (count >= 4):
            img_bg[i] = 1
            img_validcount = count
    return img_bg, img_validcount


def ShowBGMap(img_bg):
    temp = img_bg.reshape((168, 168))
    plt.figure("Image")
    plt.imshow(temp, cmap='gray')
    plt.axis('on')
    plt.title('image')
    plt.show()


def Step1(imgx, img_S, img_valid, img_bg):
    img_b = np.zeros([168*168, 3]).astype(np.float)

    for i in range(0, 168*168):
        xi=imgx[i,:]
        mi=img_valid[i,:]
        S=img_S.copy()
        for j in range(0,7):      
            if mi[j]==0:
                xi[j]=0
                S[:,j]=0
        img_b[i]=np.dot(xi,np.linalg.pinv(S))

        # temp_inv = np.linalg.pinv(np.dot(S, S.T))
        # img_b[i] = np.dot(np.dot(xi, S.T), temp_inv)
        #下面的是滤点算法二
        # if i == 5:
        #     break
    #     for j in range(0,3):
    #         if img_b[i,j]<0:
    #             img_b[i,j]=0.000000001

        # if img_bg[i] > 0:
        #     temp_s = img_S[:, img_valid[i]]
        #     temp_x = imgx[i, img_valid[i]]
        #     temp_st = temp_s.T
        #     temp_inv = np.linalg.pinv(np.dot(temp_s, temp_st))
        #     img_b[i] = np.dot(np.dot(temp_x, temp_st), temp_inv)
           

    return img_b


def render(dir, img_b, img_bg):
    imgs = np.zeros([10, 168, 168]).astype(np.float)
    imgtemp = np.zeros([10, 168*168]).astype(np.float)
    testS = np.zeros([3, 10]).astype(np.float)
    with open(dir+'/test.txt', 'r') as file:
        for i in range(0, 10):
            lines = file.readline()
            _, a, b = (float(x) for x in lines.split(','))
            testS[0][i] = math.sin(a*3.1416/180)*(math.cos(b*3.1416/180))
            testS[1][i] = math.sin(b*3.1416/180)
            testS[2][i] = math.cos(b*3.1416/180)*(math.cos(a*3.1416/180))

    
    for i in range(0, 10):
        # for j in range(0, 168*168):
            # if (img_bg[j] > 0):
            # imgtemp[i][j] = np.dot(img_b[j], testS[:, i])
            # else:
            #     imgtemp[i][j] = 0
            # if(imgtemp[i][j] < 0):
            #     imgtemp[i][j] = 0
            # if (j % 100 == 0):
                # print('Processing')
                # print(j)
                # pass
        imgs[i,:,:] = (np.dot(img_b,testS[:,i]).reshape((168, 168))).T
        '''
        with open('x'+str(i)+'.txt', "w") as f:
            for j in range(168):
                for k in range(168):
                    s = "  "
                    s +=  str(imgs[i,j,k])+' '
                    # s +=  str(b[i][0])+' '+str(b[i][1])+' '+str(b[i][2])
                    if(imgs[i,j,k]<0):
                        imgs[i,j,k]=0
                    f.write(s)
                f.write('\n')
            f.close()
        '''

    # print(testS.T)
    imgs=imgs.astype(np.uint8)
    # msvcrt.get(imgs[0])
    return imgs,testS


def shadow(Z, s,img_bg):
    sh = np.zeros([10,168,168]).astype(np.float)
    for im in range(0,10):
        a=s[0,im]
        b=s[1,im]
        c=s[2,im]
        for i in range(0, 168):
            for j in range(0, 168):
                if (img_bg[i*168+j]):
                    x0 = i
                    y0 = 167-j
                    if a > 0 and b < 0:  # 从左上侧射入
                        for m in range(x0-1, 0):  # x坐标从x0-1到0查找
                            n = b*(m-x0)/a+y0
                            if(math.ceil(n) <= 167):  # 保证向上取整得到的y坐标不超过上界
                                if(Z[m, 167-math.ceil(n)] > c*(m-x0)/a+Z[x0, y0]):
                                    sh[im,i, j] = 1
                                    break;  # 确认(x0,y0)被遮挡，跳出循环}
                                if(Z[m, 167-math.floor(n)] > c*(m-x0)/a+Z[x0, y0]):
                                    sh[im,i, j] = 1
                                    break  # 确认(x0,y0)被遮挡，跳出循环}
                            elif(math.floor(n))<=167:    #向上取整得到的y坐标超过上界但是向下取整得到的y坐标未超过上界
                    
                                if(Z[m,167-math.floor(n)]>c*(m-x0)/a+Z[x0,y0]):
                                    sh[im,i,j]=1
                                    break#确认(x0,y0)被遮挡，跳出循环}
                            else:#y超过上界，则必不会被遮挡，跳出循环
                                break

                        for n in range(y0+1,167):    #y坐标从y0+1到167查找
                            m=a*(n-y0)/b+x0
                            if(math.floor(m)>=0):#向下取整得到的x坐标不超过下界
                                if(Z[math.floor(m),167-n]>c*(n-y0)/b+Z[x0,y0]):
                                    sh[im,i,j]=1
                                    break            #确认(x0,y0)被遮挡，跳出循环}
                                if(Z[math.ceil(m),167-n]>c*(n-y0)/b+Z[x0,y0]):
                                    sh[im,i,j]=1
                                    break     #确认(x0,y0)被遮挡，跳出循环}
                    
                            elif(math.ceil(n)>=0):     #向下取整得到的x坐标超过下界但是向上取整得到的x坐标未超过下界
                    
                                if(Z[math.ceil(m),167-n]>c*(n-y0)/b+Z[x0,y0]):
                                    sh[im,i,j]=1
                                    break   #确认(x0,y0)被遮挡，跳出循环
                    
                            else:#x超过下界，则必不会被遮挡，跳出循环
                                break

                    elif a>0 and b>0:#从左下侧射入
            
                        for m in range(x0-1,0):
                            n=b*(m-x0)/a+y0
                            if(math.floor(n)>=0):   #向下取整得到的y坐标不超过下界
                    
                                if(Z[m,167-math.ceil(n)]>c*(m-x0)/a+Z[x0,y0]):
                                    sh[im,i,j]=1
                                    break      #确认(x0,y0)被遮挡，跳出循环}
                                if(Z[m,167-math.floor(n)]>c*(m-x0)/a+Z[x0,y0]):
                                    sh[im,i,j]=1
                                    break      #确认(x0,y0)被遮挡，跳出循环}
                    
                            elif(math.ceil(n)>=0):#向下取整得到的y坐标超过下界但是向上取整得到的y坐标未超过下界
                                if(Z[m,167-math.ceil(n)]>c*(m-x0)/a+Z[x0,y0]):
                                    sh[im,i,j]=1
                                    break#确认(x0,y0)被遮挡，跳出循环}
                            else:#y超过上界，则必不会被遮挡，跳出循环
                                break
            
                        for n in range(y0,0):#y坐标从y0+1到167查找
                            m=a*(n-y0)/b+x0
                            if(math.floor(m)>=0):#向下取整得到的x坐标不超过下界(即未达到xy坐标系的左边界)
                    
                                if(Z[math.floor(m),167-n]>c*(n-y0)/b+Z[x0,y0]):
                                    sh[im,i,j]=1
                                    break    #确认(x0,y0)被遮挡，跳出循环}
                                if(Z[math.ceil(m),167-n]>c*(n-y0)/b+Z[x0,y0]):
                                    sh[im,i,j]=1
                                    break #确认(x0,y0)被遮挡，跳出循环}
                    
                            elif(math.ceil(n)>=0):#向下取整得到的x坐标超过下界但是向上取整得到的x坐标未超过下界
                    
                                if(Z[math.ceil(m),167-n]>c*(n-y0)/b+Z[x0,y0]):
                                    sh[im,i,j]=1
                                    break#确认(x0,y0)被遮挡，跳出循环
                    
                            else:#x超过下界，则必不会被遮挡，跳出循环
                                break

                    elif a<0 and b<0:#从右上侧射入
            
                        for m in range(x0+1,167):
                            n=b*(m-x0)/a+y0
                            if(math.ceil(n)<=167):#保证向上取整得到的y坐标不超过上界
                    
                                if(Z[m,167-math.ceil(n)]>c*(m-x0)/a+Z[x0,y0]):
                                    sh[im,i,j]=1
                                    break#确认(x0,y0)被遮挡，跳出循环}
                                if(Z[m,167-math.floor(n)]>c*(m-x0)/a+Z[x0,y0]):
                                    sh[im,i,j]=1
                                    break#确认(x0,y0)被遮挡，跳出循环}
                    
                            elif(math.floor(n)<=167):#向上取整得到的y坐标超过上界但是向下取整得到的y坐标未超过上界
                    
                                if(Z[m,167-math.floor(n)]>c*(m-x0)/a+Z[x0,y0]):
                                    sh[im,i,j]=1
                                    break #确认(x0,y0)被遮挡，跳出循环}
                    
                            else:#y超过上界，则必不会被遮挡，跳出循环
                                break

                        for n in range(y0+1,167):#y坐标从y0-1到0查找
                            m=a*(n-y0)/b+x0
                            if(math.ceil(m)<=167):#向上取整得到的x坐标不超过上界
                    
                                if(Z[math.ceil(m),167-n]>c*(n-y0)/b+Z[x0,y0]):
                                    sh[im,i,j]=1
                                    break#确认(x0,y0)被遮挡，跳出循环}
                                if(Z[math.floor(m),167-n]>c*(n-y0)/b+Z[x0,y0]):
                                    sh[im,i,j]=1
                                    break#确认(x0,y0)被遮挡，跳出循环}
                    
                            elif(math.floor(m)<=167):#向上取整得到的x坐标超过上界但是向下取整得到的x坐标未超过上界
                    
                                if(Z[math.floor(m),167-n]>c*(n-y0)/b+Z[x0,y0]):
                                    sh[im,i,j]=1
                                    break#确认(x0,y0)被遮挡，跳出循环
                    
                            else:   #x超过上界，则必不会被遮挡，跳出循环
                                break

                    else:#从右下侧侧射入
            
                        for m in range(x0+1,167):
                            n=b*(m-x0)/a+y0
                            if n<167 and n>0:
                                if(math.floor(n)>=0):#向下取整得到的y坐标不超过下界
                        
                                    if(Z[m,167-math.ceil(n)]>c*(m-x0)/a+Z[x0,y0]):
                                        sh[im,i,j]=1
                                        break #确认(x0,y0)被遮挡，跳出循环}
                                    if(Z[m,167-math.floor(n)]>c*(m-x0)/a+Z[x0,y0]):
                                        sh[im,i,j]=1
                                        break#确认(x0,y0)被遮挡，跳出循环}
                        
                                elif(math.ceil(n)<=167):#向下取整得到的y坐标超过下界但是向上取整得到的y坐标未超过下界
                        
                                    if(Z[m,167-math.ceil(n)]>c*(m-x0)/a+Z[x0,y0]):
                                        sh[im,i,j]=1
                                        break   #确认(x0,y0)被遮挡，跳出循环}
                        
                                else:    #y超过上界，则必不会被遮挡，跳出循环
                                    break
            
                        for n in range(y0,0):#y坐标从y0+1到167查找
                            m=a*(n-y0)/(b-0.00001)+x0
                            if m<0:
                                continue
                            if(math.ceil(m)<=167): #向上取整得到的x坐标不超过上界
                                if(Z[math.ceil(m),167-n]>c*(n-y0)/b+Z[x0,y0]):
                                    sh[im,i,j]=1
                                    break   #确认(x0,y0)被遮挡，跳出循环}
                                if(Z[math.floor(m),167-n]>c*(n-y0)/b+Z[x0,y0]):
                                    sh[im,i,j]=1
                                    break   #确认(x0,y0)被遮挡，跳出循环}
                    
                            elif(math.floor(n)<=167):#向上取整得到的x坐标超过下界但是向下取整得到的x坐标未超过上界
                    
                                if(Z[math.floor(m),167-n]>c*(n-y0)/b+Z[x0,y0]):
                                    sh[im,i,j]=1
                                    break#确认(x0,y0)被遮挡，跳出循环
                    
                            else:#x超过上界，则必不会被遮挡，跳出循环
                                break 
    for im in range(0,10):
        for i in range(1,167):
            for j in range(1,167):
                if(sh[im,i,j]==1) and (sh[im,i-1,j]==0 or sh[im,i+1,j]==0 or sh[im,i,j-1]==0 or sh[im,i,j+1]==0): #初步判断是否在阴影边缘
                    sh[im,i,j]=0.5
    return sh


def show3d(z):
    
    x = np.linspace(0,167,168)
    y = np.linspace(0,167,168)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(200, 200))
    ax = Axes3D(fig)
    surf = ax.plot_surface(X, Y, z,
                       rstride=1, 
                       cstride=1, 
                       cmap=plt.get_cmap('rainbow'))  
    ax.set_zlim(-50, 300)
    plt.title("3D")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def OptimizeZ(Z):

    # 对称优化
    # ZZ=Z.copy()
    # for i in range(0,84):
    #     ZZ[:,i]=Z[:,167-i]
    #     ZZ[:,167-i]=Z[:,i]
    # Z=(Z+ZZ)/2
    #  Z=Z*1.2
    frontpoint=np.max(Z)
    backpoint=np.min(Z)
    depth=180
    scale=depth/(frontpoint-backpoint)
    Z=Z*scale
    return Z


def inter(b):
    h = w = 168

    b = np.array([[b[i, j] if b[i, j] != 0 or j != 3 else 0.00000000000000000001 for j in range(3)] for i in range(w * h)])

    b = np.array([[b[i, j] if b[i, j] != 0 or j == 3 else 0.00000000000000000000000000000000000000000000000000000000000001 for j in range(3)] for i in range(w * h)])

    z_x = np.reshape(-b[:, 0] / b[:, 2] , (h, w))
    z_y = np.reshape(-b[:, 1] / b[:, 2], (h, w))
    z_x=z_x.T
    z_y=z_y.T

    max = 1
    lam = 1
    u = 3

    for k in range(1, w - 1):
        for l in range(1, h - 1):
            if abs(z_x[l, k])>max or abs(z_y[l, k]) > max:
                z_x[l, k] = (z_x[l - 1, k] + z_x[l + 1, k] + z_x[l, k + 1] + z_x[l, k - 1]) / 4
                z_y[l, k] = (z_y[l - 1, k] + z_y[l + 1, k] + z_y[l, k + 1] + z_y[l, k - 1]) / 4
    
    zz_x = np.zeros((h*2, w*2))
    zz_x[0 : h, 0 : w] = z_x[:, :]
    zz_x[h : 2 * h, 0 : w] = z_x[h - 1 : : -1]
    zz_x[:, w : w * 2] = zz_x[:, w - 1 : : -1]
    zz_y = np.zeros((h*2,w*2))
    zz_y[0 : h, 0 : w] = z_y[:, :]
    zz_y[h : 2 * w, 0 : w] = z_y[h - 1 : : -1]
    zz_y[:, w : w * 2] = zz_y[:, w - 1 : : -1]
    z_x = zz_x
    z_y = zz_y
    # print(z_y.size)
    
    h = h * 2
    w = w * 2

    for i in range(1, w - 1):
        for j in range(1, h - 1):
            if abs(z_x[j, i]) > max or abs(z_y[j, i]) > max:
                z_x[j, i] = (z_x[j - 1, i] + z_x[j + 1, i] + z_x[j, i + 1] + z_x[j, i - 1]) / 4
                z_y[j, i] = (z_y[j - 1, i] + z_y[j + 1, i] + z_y[j, i + 1] + z_y[j, i - 1]) / 4

    C_x = np.fft.fft2(z_x)
    C_y = np.fft.fft2(z_y)

    C = np.zeros((h, w)).astype('complex')
    C_xx = np.zeros((h, w)).astype('complex')
    C_yy = np.zeros((h, w)).astype('complex')

    for m in range(w):
        for n in range(h):
            wx = 2 * pi * m / w
            wy = 2 * pi * n/ h
            if sin(wx) == 0 and sin(wy) == 0:
                C[n, m] = 0
            else:
                cons=(1 + lam) * (sin(wx) *sin(wx) + sin(wy) *sin(wy)) + u * (sin(wx) *sin(wx) + sin(wy) *sin(wy)) ** 2
                C[n, m]=(C_x[n, m] * (complex(0, -1) * sin(wx)) + C_y[n, m] * (complex(0, -1) * sin(wy))) / cons
            C_xx[n, m] = complex(0, 1) * sin(wx) * C[n, m]
            C_yy[n, m] = complex(0, 1) * sin(wy) * C[n, m]

    

    h = h // 2
    w = w // 2
    Z = np.fft.ifft2(C).real
    Z = Z[0 : h, 0 : w]

    Z_xx = np.fft.ifft2(C_xx).real
    Z_yy = np.fft.ifft2(C_yy).real
    
    Z_xx = Z_xx[0 : h, 0 : w]
    Z_yy = Z_yy[0 : h, 0 : w]

    for i in range(1, w - 1):
        for j in range(1, h - 1):
            if abs(Z_xx[j, i]) > max or abs(Z_yy[j, i]) > max:
                Z_xx[j, i] = (Z_xx[j - 1, i] + Z_xx[j + 1, i] + Z_xx[j, i + 1] + Z_xx[j, i - 1]) / 4
                Z_yy[j, i] = (Z_yy[j - 1, i] + Z_yy[j + 1, i] + Z_yy[j, i + 1] + Z_yy[j, i - 1]) / 4
                Z[j, i] = (Z[j - 1, i] + Z[j + 1, i] + Z[j, i + 1] + Z[j, i - 1]) / 4
    

    # print(Z)

    return Z

def Redefine(B1):
    B=B1.copy()
    B2=np.zeros([168*168,3])
    B2[:,0]=B[:,1]
    B2[:,1]=B[:,2]
    B2[:,2]=np.abs(B[:,0])
    return B2

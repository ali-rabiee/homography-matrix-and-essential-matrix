'''
Ali Rabiee 99210389
Practical HW2
Run the code and after showing any result 
please press a key to show next result
'''
import numpy as np
import cv2
from sklearn.metrics import mean_squared_error

'''Practice3'''

#read images
img1 = cv2.imread('1.jpg')
img2 = cv2.imread('2.jpg')
img3 = cv2.imread('3.jpg')

#corners of image1
corners1 = np.array([[50, 44], 
                     [321, 187], 
                     [62, 765], 
                     [323, 602]])
#corners of new image1
new_corners1 = np.array([[0, 0],
                      [img1.shape[1], 0],
                      [0, img1.shape[0]],
                      [img1.shape[1], img1.shape[0]]])  

#calculate homography matrix without employing python packages 
def FourPoint(x1, x2):
    A = []
    for i in range(len(x1)):
      A.append([-x1[i][0],-x1[i][1],-1,0,0,0,x2[i][0]*x1[i][0],x2[i][0]*x1[i][1],x2[i][0]])
      A.append([0,0,0,-x1[i][0],-x1[i][1],-1,x2[i][1]*x1[i][0],x2[i][1]*x1[i][1],x2[i][1]])
    U, sigma, V = np.linalg.svd(A)
    V = V.transpose()
    H = V[:,8]
    H = np.array(H)
    H = H.reshape(3,3)
    return H / H[2,2]

H1 = FourPoint(corners1, new_corners1)
        
#calculate homography matrix with employing python packages
h1, s1 = cv2.findHomography(corners1, new_corners1)

#show output1
out1 = cv2.warpPerspective(img1, h1, (img1.shape[1], img1.shape[0]))
cv2.imshow('Output image1', out1)
cv2.waitKey()
cv2.destroyAllWindows()

#corners of image2
corners2 = np.array([[0, 0],
                     [img2.shape[1], 0],
                     [img2.shape[1], img2.shape[0]],
                     [0, img2.shape[0]]])

#corners of image3
corners3 = np.array([[240, 250],
                     [1176, 173],
                     [1175, 520],
                     [242, 482]])

#calculate homography matrix without employing python packages 
H2 = FourPoint(corners2, corners3)
   
#calculate homography matrix with employing python packages 
h2, s2 = cv2.findHomography(corners2, corners3)

#show output2 without employing python packages

im_src  = cv2.resize(img2,(1500, 538), interpolation = cv2.INTER_AREA)
im_dst = cv2.imread('3.jpg')
for i in range(im_src.shape[1]):
   for j in range(im_src.shape[0]):
      k = np.matmul(H2,[[i],[j],[1]])
      k = k / k[2];
      k = [int(i) for i in k]
      try:
          im_dst[k[1]][k[0]] = im_src[j][i]
      except:
          pass
cv2.imshow('Output image2 without package', im_dst)
cv2.waitKey()
cv2.destroyAllWindows()     
#show output2 with employing python packages
out2 = cv2.warpPerspective(img2, h2, (img3.shape[1], img3.shape[0]))
wall = img3.copy()
cv2.fillConvexPoly(wall, corners3, 0, 16)
out2 = wall + out2
cv2.imshow('Output image2 with package', out2)
cv2.waitKey()
cv2.destroyAllWindows()


'''Practice4'''

'''Part a'''
print('Part a')
#Correspondence points in homogenous coordinates    
x1 = np.array([[242, 84, 1], [458, 40, 1], [246, 280, 1],[457, 323, 1],
              [332, 291, 1], [49, 283, 1], [165, 333, 1],[337, 197, 1]])
x2 = np.array([[130, 33, 1], [504, 33, 1], [128, 260, 1],[506, 258, 1],
              [316, 34, 1], [317, 257, 1], [60, 305, 1],[575, 309, 1]])
x3 = np.array([[25, 7, 1], [239, 53, 1], [25, 290, 1],[242, 250, 1],
              [152, 40, 1], [154, 264, 1], [315, 300, 1],[440, 255, 1]])

#Correspondence points in 2D coordinates
p1 = np.array([[242, 84], [458, 40], [246, 280],[457, 323],
              [332, 291], [49, 283], [165, 333],[337, 197]])
p2 = np.array([[130, 33], [504, 33], [128, 260],[506, 258],
              [316, 34], [317, 257], [60, 305],[575, 309]])
p3 = np.array([[25, 7], [239, 53], [25, 290],[242, 250],
              [152, 40], [154, 264], [315, 300],[440, 255]])

#Find the essential matrix without employing packages
def EightPoint(x1, x2):
    X = np.zeros((8, 9))
    for i in range(8):
        X[i] = [x1[i, 0]*x2[i, 0],  x1[i, 0]*x2[i, 1],  x1[i, 0]*x2[i, 2],
                x1[i, 1]*x2[i, 0],  x1[i, 1]*x2[i, 1],  x1[i, 1]*x2[i, 2],
                x1[i, 2]*x2[i, 0],  x1[i, 2]*x2[i, 1],  x1[i, 2]*x2[i, 2]]
        
    X = np.matmul(X.transpose(), X)
    eigenValues, eigenVectors = np.linalg.eig(X)
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    #solution of X*E = 0 is F:
    F = eigenVectors[8].reshape(3,3)
    U, sigma, V = np.linalg.svd(F)
    S = np.zeros((3,3))
    S[0, 0] = S[1, 1] = (sigma[0] + sigma[1])/2
    E = np.matmul((np.matmul(U, S)), V.transpose())
    return E

#Calculate Essential matrices
E12 = EightPoint(x1, x2)
E23 = EightPoint(x2, x3)
E13 = EightPoint(x1, x3)

#normalize
E12 = E12 / E12[2, 2]
E23 = E23 / E23[2, 2]
E13 = E13 / E13[2, 2]

print('Essential Matrices without employing packages\n')
print('E12')
print(E12)
print('\n')
print('E23')
print(E23)
print('\n')
print('E13')
print(E13)
print('\n')

'''part b'''
print('\nPart b')
#Calculate R & T from essential matrices without employing python packages
R12, r12, T12 = cv2.decomposeEssentialMat(E12)
R23, r23, T23 = cv2.decomposeEssentialMat(E23)

#transforming 1 to 3 directly
R13, r13, T13 = cv2.decomposeEssentialMat(E13)

#transforming 1 to 3 undirectly
R = np.matmul(r12, R23)
T = np.matmul(r12, T23) + T12
T = T / np.linalg.norm(T)

print('Transformation directly from 1 to 3:\n')
print('R =')
print(r13)
print('')
print('T =')
print(T13)
print('\nTransformation undirectly from 1 to 3:\n')
print('R =')
print(R)
print('')
print('T =')
print(T)

#transforming 3 to 1 directly
R31 = r13.transpose()
T31 = -1*np.matmul(r13.transpose(), T13)

#transforming 3 to 1 undirectly
r31 = R.transpose()
t31 = -1*np.matmul(R.transpose(), T)

print('\nTransformation directly from 3 to 1:\n')
print('R =')
print(R31)
print('')
print('T =')
print(T31)
print('\nTransformation undirectly from 3 to 1:\n')
print('R =')
print(r31)
print('')
print('T =')
print(t31)

'''Part c'''
#Find the essential matrices with employing packages
e12, mask = cv2.findEssentialMat(p1, p2)
e23, mask = cv2.findEssentialMat(p2, p3)
e13, mask = cv2.findEssentialMat(p1, p3)

print('\nPart c')
print('Essential Matrix with employing packages\n')
print('e12')
print(e12)
print('\n')
print('e23')
print(e23)
print('\n')
print('e13')
print(e13)
print('\n')

'''Part d'''
#Calculate R & T from essential matrices with employing python packages
R12, r12, T12 = cv2.decomposeEssentialMat(e12)
R23, r23, T23 = cv2.decomposeEssentialMat(e23)

#transforming 1 to 3 directly
R13, r13, T13 = cv2.decomposeEssentialMat(e13)

#transforming 1 to 3 undirectly
R = np.matmul(r12, R23)
T = np.matmul(r12, T23) + T12
T = T / np.linalg.norm(T)

print('\nPart d')
print('Transformation directly from 1 to 3:\n')
print('R =')
print(r13)
print('')
print('T =')
print(T13)
print('\nTransformation undirectly from 1 to 3:\n')
print('R =')
print(R)
print('')
print('T =')
print(T)

#transforming 3 to 1 directly
R31 = r13.transpose()
T31 = -1*np.matmul(r13.transpose(), T13)

#transforming 3 to 1 undirectly
r31 = R.transpose()
t31 = -1*np.matmul(R.transpose(), T)

print('\nTransformation directly from 3 to 1:\n')
print('R =')
print(R31)
print('')
print('T =')
print(T31)
print('\nTransformation undirectly from 3 to 1:\n')
print('R =')
print(r31)
print('')
print('T =')
print(t31)

'''Part e'''
print('\nPart e')
#calculate MSE
print(f'MSE of 1-2 = {mean_squared_error(E12,e12) }')
print(f'MSE of 2-3 = {mean_squared_error(E23,e23) }')
print(f'MSE of 1-3 = {mean_squared_error(E13,e13) }') 

































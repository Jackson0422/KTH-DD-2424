import copy
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tqdm
from torch_gradient_computations import ComputeGradsWithTorch
import time
# ==================================================================================== Exercise 1.1 function ====================================================================================
def separator(filename):
    cifar_dir = r'C:\Users\JKX\Desktop\DL\Assignment1\dataset'            # set the file address
    with open(cifar_dir + filename, 'rb') as fo:                          # address + filename = target file name
        dict = pickle.load(fo, encoding='bytes')                          # use the dictionary to store the target file
    X = dict[b'data'].astype(np.float64) / 255.0                          # let the image data between 0 and 1
    y = dict[b'labels']                                                   # get the label
    Y = np.eye(10)[y]                                                     # transfer the label to one-hot style
    return X.T, Y.T, y                                                    # return the result

# ==================================================================================== test function in 1.1 ====================================================================================
filename_train = '\data_batch_1'
filename_val = '\data_batch_2'
filename_test = r'\test_batch'
X_train, Y_train, y_train = separator(filename_train) # X_train 10000 * 3027  10000 * 10

# ==================================================================================== Exercise1.2 =============================================================================================
mean_X_train = np.mean(X_train, axis=1).reshape(3072, 1)       # the mean of the training data
std_X_train = np.std(X_train, axis=1).reshape(3072, 1)         # the standard of the training data
X_train_after = (X_train - mean_X_train) / std_X_train          # normalize the training data

# use the training data's mean and standard deviation vector to normalize the validation data
X_val, Y_val, y_val = separator(filename_val)                    # X_val 10000 * 3027
X_val_after = (X_val - mean_X_train) / std_X_train               # normalize the validation data   3072 * 10000

# use the training data's mean and standard deviation vector to normalize the test data
X_test, Y_test, y_test = separator(filename_test)                # X_test 10000 * 3027
X_test_after = (X_test - mean_X_train) / std_X_train             # normalize the test data 3072 * 10000

# ==================================================================================== Exercise1.3 =============================================================================================
rng = np.random.default_rng()                                      # creat a random number maker
BitGen = type(rng.bit_generator)                                   # get the type of the random number maker
seed = 42                                                          # set the seed of the random number seed
rng.bit_generator.state = BitGen(seed).state                       # create a bite generator
init_net = {}                                                      # initial the weigght and bias dictionary
init_net['W'] = 0.01 * rng.standard_normal(size = (10 , 3072))     # make the weight randomly             10 * 3072
init_net['b'] = np.zeros((10 , 1))                                 # make the bias randomly               10 * 1

# ==================================================================================== Exercise1.4 =============================================================================================
def ApplyNetwork(X, network):
    S = network['W'].dot(X) + network['b']                         # line is different class's score and clomn is different photo    10 * 10000
    exp_s = np.exp(S)                                              # 10 * 100000
    sum_exp = np.sum(exp_s, axis = 0)                              # 10 * 1
    P = exp_s / sum_exp                                            # 10 * 10000
    return P                                                       # get a probabilties for every photo in every class


# ==================================================================================== Exercise1.5 =============================================================================================
def loss_function(P, y):                                     # the input is probabilities and label
    loss = -y * np.log(P)
    return loss                                              # 10 * 10000 get a loss value

def compute_loss(P, y,x, lam, network):                      # calculate the loss with regulaztion
    loss = loss_function(P, y)
    L = 1 / x.shape[1] * np.sum(loss) + lam * np.sum((network['W']**2))
    return L                                                 # 10 * 1


# ==================================================================================== Exercise1.6 =============================================================================================
def ComputeAccuracy(P, y):
    P_argmax = np.argmax(P, axis = 0)              # 10000 * 1
    y_1 = np.argmax(y, axis=0)
    acc_rate = np.equal(P_argmax, y_1).sum() / 10000       # get the right prediction results
    print("The least accuracy in test is: ", acc_rate)
    return acc_rate


# ==================================================================================== Exercise1.7 =============================================================================================
def Backwardpass(X, Y, P, network, lam):           # one-hot，， Wb，λ
    # 计算交叉熵损失的梯度
    # Y = Y.T
    dS = -(Y - P)
    N = X.shape[1]
    W = network['W']
    dW = (1 / N) *np.dot(dS, X.T) + 2 * lam * W    # 10 * 10  #  Ｗ
    db = (1 / N) * np.sum(dS, axis=1, keepdims=True) # 10 * 1　#　ｂ


    grads = {}
    grads['dW'] = dW
    grads['db'] = db
    return grads

def relative_error(g_n,g_a):
    return np.abs(np.sum(g_n-g_a))

# ==================================================================================== Exercise1.8 =============================================================================================
rng = np.random.default_rng(seed = 42)
def MiniBatchGD(X, Y, GDparams, init_net, lam):
    train_loss_list = []
    trained_net = copy.deepcopy(init_net)
    n = X.shape[1]

    for i in range(GDparams['n_epochs']):
        random = np.random.permutation(n)

        for j in range(10000 // GDparams['n_batch']):
            j_start = j * GDparams['n_batch']
            j_end = (j + 1) * GDparams['n_batch']
            inds = random[j_start:j_end]
            Xbatch = X[:, inds]
            Ybatch = Y[:, inds]
            P_1 = ApplyNetwork(Xbatch, trained_net)
            grads_my = Backwardpass(Xbatch, Ybatch, P_1, trained_net, lam)
            trained_net['W'] -= GDparams['eta'] * grads_my['dW']
            trained_net['b'] -= GDparams['eta'] * grads_my['db']

        P_2 = ApplyNetwork(X, trained_net)
        loss_epoch = compute_loss(P_2, Y, X, lam, trained_net)
        train_loss_list.append(loss_epoch)
        # print("train_loss_list_sum: ", train_loss_list)
    return trained_net, train_loss_list

GDparams = {}
GDparams['n_batch'] = 100
GDparams['eta'] = 0.001
GDparams['n_epochs'] = 40
lam_1 = 1
train_net, loss_list_train = MiniBatchGD(X_train_after, Y_train, GDparams, init_net, lam_1)
print("loss_list_train's first number is: ", loss_list_train[0])
trained_net , loss_list_val= MiniBatchGD(X_val_after, Y_val, GDparams, init_net, lam_1)
plt.plot(loss_list_train, color="Green", label="Training Loss")
plt.plot(loss_list_val, color="red", label="Validation Loss")
plt.legend(loc='upper right')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
P_3 = ApplyNetwork(X_test_after, train_net)
acc_end = ComputeAccuracy(P_3, Y_test)

# 显示权重图
Ws = trained_net['W'].transpose().reshape((32, 32, 3, 10), order='F')
W_im = np.transpose(Ws, (1, 0, 2, 3))
for i in range(10):
    w_im = W_im[:, :, :, i]
    w_im_norm = (w_im - np.min(w_im)) / (np.max(w_im) - np.min(w_im))
    plt.imshow(w_im_norm)
    plt.show()
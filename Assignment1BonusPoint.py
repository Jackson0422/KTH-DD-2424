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
mean_X_train = np.mean(X_train, axis=1).reshape(3072, 1)       # the mean of the training data         3072 * 1
std_X_train = np.std(X_train, axis=1).reshape(3072, 1)         # the standard of the training data    3072 * 1
X_train_after = (X_train - mean_X_train) / std_X_train          # normalize the training data            10000 * 3027

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
init_net['W'] = 0.01 * rng.standard_normal(size = (10 , 3072))     # make the weight randomly                  10 * 3072
init_net['b'] = np.zeros((10 , 1))                                 # make the bias randomly                   10 * 1

# ==================================================================================== Exercise1.4 =============================================================================================
def ApplyNetwork(X, network):
    S = network['W'].dot(X) + network['b']                         # line is different class's score and clomn is different photo    10 * 10000
    # exp_s = np.exp(S)                                              # 10 * 100000
    # sum_exp = np.sum(exp_s, axis = 0)                              # 10 * 1
    # P = exp_s / sum_exp                                            # 10 * 10000

    # bonus 2.2
    P = 1 / (1 + np.exp(-S) )                                      # Sigmoid
    return P                                                       # get a probabilties for every photo in every class


# ==================================================================================== Exercise1.5 =============================================================================================
def loss_function(P, y):                                     # the input is probabilities and label
    # loss = -y * np.log(P)

    # Bonus 2.2
    loss =  -np.mean((1-y)* np.log(1 - P)+ y* np.log(P), axis=0)
    return loss.mean()                                              # 10 * 10000 get a loss value

def compute_loss(P, y,x, lam, network):                      # calculate the loss with regulaztion
    # loss = loss_function(P, y)
    # L = loss + lam * np.sum((network['W']**2))

    # Bonus 2.2
    BCE_loss =  loss_function(P, y)
    reg = lam * np.sum(network['W'] ** 2)
    L = BCE_loss + reg
    return L, BCE_loss                                                  # 10 * 1


# ==================================================================================== Exercise1.6 =============================================================================================
def ComputeAccuracy(P, y):
    P_argmax = np.argmax(P, axis = 0)              # 10000 * 1
    y_1 = np.argmax(y, axis=0)
    acc_rate = np.equal(P_argmax, y_1).sum() / y.shape[1]       # get the right prediction results
    print("The least accuracy in test is: ", acc_rate)
    return acc_rate


# ==================================================================================== Exercise1.7 =============================================================================================
def Backwardpass(X, Y, P, network, lam):
    # Y = Y.T
    # dS = -(Y - P)
    # N = X.shape[1]
    # W = network['W']
    # dW = (1 / N) *np.dot(dS, X.T) + 2 * lam * W    # 10 * 10   Ｗ
    # db = (1 / N) * np.sum(dS, axis=1, keepdims=True) # 10 * 1　　ｂ

    # Bonus 2.2
    K = X.shape[0]
    N = X.shape[1]
    W = network['W']
    dS = (P - Y) / K
    dW = (1 / N) *np.dot(dS, X.T) + 2 * lam * W
    db = (1 / N) * np.sum(dS, axis=1, keepdims=True)

    grads = {}
    grads['dW'] = dW
    grads['db'] = db
    return grads

def relative_error(g_n,g_a):
    return np.abs(np.sum(g_n-g_a))

# ==================================================================================== Exercise1.8 =============================================================================================
rng = np.random.default_rng(seed = 42)
def MiniBatchGD(X, Y, GDparams, init_net, lam, X_val, Y_val):
    train_loss_list = []
    val_loss_list = []
    train_cost_list = []
    val_cost_list = []
    trained_net = copy.deepcopy(init_net)
    n = X.shape[1]
    init_eta = GDparams['eta']
    for i in range(GDparams['n_epochs']):
        random = np.random.permutation(n)
        Current_eta = init_eta * (0.1 **(i // 10))
        for j in range(X.shape[1] // GDparams['n_batch']):
            j_start = j * GDparams['n_batch']
            j_end = (j + 1) * GDparams['n_batch']
            inds = random[j_start:j_end]
            Xbatch = X[:, inds]
            Ybatch = Y[:, inds]
            Xbatch_flip = random_flip(Xbatch)
            P_1 = ApplyNetwork(Xbatch_flip, trained_net)
            grads_my = Backwardpass(Xbatch_flip, Ybatch, P_1, trained_net, lam)
            trained_net['W'] -= Current_eta * grads_my['dW']
            trained_net['b'] -= Current_eta * grads_my['db']

        P_2 = ApplyNetwork(X, trained_net)
        cost_train, loss_train = compute_loss(P_2, Y, X, lam, trained_net)
        train_cost_list.append(cost_train)
        train_loss_list.append(loss_train)
        P_3 = ApplyNetwork(X_val, trained_net)
        val_cost, val_loss = compute_loss(P_3, Y_val, X_val, lam, trained_net)
        val_cost_list.append(val_cost)
        val_loss_list.append(val_loss)
        # print("train_loss_list_sum: ", train_loss_list)
    return trained_net, train_loss_list, val_loss_list, train_cost_list, val_cost_list

# GDparams = {}
# GDparams['n_batch'] = 100
# GDparams['eta'] = 0.001
# GDparams['n_epochs'] = 40
# lam_1 = 1
# train_net, loss_list_train = MiniBatchGD(X_train_after, Y_train, GDparams, init_net, lam_1)
# print("loss_list_train's first number is: ", loss_list_train[0])
# trained_net , loss_list_val= MiniBatchGD(X_val_after, Y_val, GDparams, init_net, lam_1)
# plt.plot(loss_list_train, color="Green", label="Training Loss")
# plt.plot(loss_list_val, color="red", label="Validation Loss")
# plt.legend(loc='upper right')
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.show()
# P_3 = ApplyNetwork(X_test_after, train_net)
# acc_end = ComputeAccuracy(P_3, Y_test)

# Ws = trained_net['W'].transpose().reshape((32, 32, 3, 10), order='F')
# W_im = np.transpose(Ws, (1, 0, 2, 3))
# for i in range(10):
#     w_im = W_im[:, :, :, i]
#     w_im_norm = (w_im - np.min(w_im)) / (np.max(w_im) - np.min(w_im))
#     plt.imshow(w_im_norm)
#     plt.show()

# ==================================================================================== Exercise 2.1 =============================================================================================
# 2.1 a
batch1_name = '\data_batch_1'
batch2_name = '\data_batch_2'
batch3_name = '\data_batch_3'
batch4_name = '\data_batch_4'
batch5_name = '\data_batch_5'
batch1_train, batch1_label_onehot, batch1_label_true = separator(batch1_name)
batch2_train, batch2_label_onehot, batch2_label_true = separator(batch2_name)
batch3_train, batch3_label_onehot, batch3_label_true = separator(batch3_name)
batch4_train, batch4_label_onehot, batch4_label_true = separator(batch4_name)
batch5_train, batch5_label_onehot, batch5_label_true = separator(batch5_name)
all_data_train = np.hstack((batch1_train, batch2_train, batch3_train, batch4_train, batch5_train))  # 3072 * 50000
all_data_label_onehot = np.hstack((batch1_label_onehot, batch2_label_onehot, batch3_label_onehot, batch4_label_onehot, batch5_label_onehot)) # 10 * 50000
train_data = all_data_train[:, :49000]               # 3072 * 49000
val_data = all_data_train[:, 49000:]                 # 3072 * 1000
train_onehot = all_data_label_onehot[:, :49000]    # 10 * 49000
val_onehot = all_data_label_onehot[:, 49000:]      # 10 * 1000
mean_train_data = np.mean(train_data, axis=1).reshape(3072, 1)              # the mean of the training data         3072 * 1
std_train_data = np.std(train_data, axis=1).reshape(3072, 1)                # the standard of the training data    3072 * 1
train_data_after = (train_data - mean_train_data) / std_train_data          # normalize the training data           10000 * 3027

# 2.1 (b)
def random_flip(data, p = 0.5):
    X_flipped = data.copy()
    N = data.shape[1]

    aa = np.int32(np.arange(32)).reshape((32, 1))
    bb = np.int32(np.arange(31, -1, -1)).reshape((32, 1))
    vv = np.tile(32 * aa, (1, 32))
    ind_flip = vv.reshape((32 * 32, 1)) + np.tile(bb, (32, 1))
    inds_flip = np.vstack((ind_flip, 1024 + ind_flip))
    inds_flip = np.vstack((inds_flip, 2048 + ind_flip))
    inds_flip = inds_flip.squeeze()

    for i in range(N):
        if np.random.rand() < p:
            X_flipped[:, i] = X_flipped[inds_flip, i]

    return X_flipped

def plot_ground_truth_histogram(X_test, Y_test, net, title_prefix=""):

    P = ApplyNetwork(X_test, net)

    true_labels = np.argmax(Y_test, axis=0)
    pred_labels = np.argmax(P, axis=0)
    true_class_probs = P[true_labels, np.arange(P.shape[1])]


    correct_probs = true_class_probs[pred_labels == true_labels]
    incorrect_probs = true_class_probs[pred_labels != true_labels]


    plt.figure()
    plt.hist(correct_probs, bins=20, alpha=0.7, label="Correctly Classified")
    plt.hist(incorrect_probs, bins=20, alpha=0.7, label="Incorrectly Classified")
    plt.xlabel("Probability for Ground Truth Class")
    plt.ylabel("Number of Examples")
    plt.title(title_prefix + " Histogram of Ground Truth Class Probabilities")
    plt.legend(loc="upper right")
    plt.show()

if __name__ == "__main__":
    # eta_list =[0.001, 0.01]
    # L2_list = [0.01, 0.001, 0.1]
    # Batch_size_list = [100, 100, 500]
    # acc_list = []
    # acc_list_train = []
    # for i in range(3):
    #     GDparams = {}
    #     GDparams['n_batch'] = Batch_size_list[i]
    #     for k in range(2):
    #         GDparams['eta'] = eta_list[k]
    #         for j in range(3):
    #             lam_1 = L2_list[j]
    #             GDparams['n_epochs'] = 40
    #             train_net, loss_list_train = MiniBatchGD(train_data_after, train_onehot, GDparams, init_net, lam_1)
    #             P_3 = ApplyNetwork(X_test_after, train_net)
    #             acc_end = ComputeAccuracy(P_3, Y_test)
    #             acc_list.append(acc_end)
    #
    # print("The test accuracy list is: ", acc_list)
    # # print("The train accuracy list is: ", acc_list_train)
    # plt.plot(acc_list, color="Green", label="Test Accuracy")
    # # plt.plot(acc_list_train, color="Blue", label="Train Accuracy")
    # plt.legend(loc='upper right')
    # plt.xlabel("Times")
    # plt.ylabel("Accuracy")
    # plt.show()

    GDparams = {}
    GDparams['n_batch'] = 100
    GDparams['eta'] = 0.01
    GDparams['n_epochs'] = 40
    lam_1 = 0.1
    train_net, loss_list_train, val_lost_list, train_cost_list, val_cost_list = MiniBatchGD(train_data_after,train_onehot, GDparams,init_net, lam_1, val_data,val_onehot)
    plt.figure()
    plt.plot(loss_list_train, color="Green", label="Training Loss")
    plt.plot(val_lost_list, color="red", label="Validation Loss")
    plt.legend(loc='upper right')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.show()

    plt.figure()  # 创建另一个新的图形窗口
    plt.plot(train_cost_list, color="Green", label="Training Cost")
    plt.plot(val_cost_list, color="red", label="Validation Cost")
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.title("Cost Curve")
    plt.legend(loc='upper right')
    plt.show()

    # plot_ground_truth_histogram(X_test_after, Y_test, train_net, 'new_training')
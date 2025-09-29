import numpy as np
from GL import ComputeGradsWithTorch
import pickle
import tqdm
import copy
import matplotlib.pyplot as plt

# Exercise 1
def conv_res(X, filters):
    n_img = X.shape[-1]
    n_filter = filters.shape[-1]
    f_size = filters.shape[0]
    conv_result = np.zeros((32 // f_size, 32 // f_size, n_filter, n_img))

    for i in range(n_img):
        for j in range(n_filter):
            for k in range( 32 // f_size):
                for x in range(32 // f_size):
                    patch = X[k * f_size:(k + 1) * f_size, x * f_size:(x + 1) * f_size, :, i]
                    conv_result[k, x, j, i] = np.sum(np.multiply(patch, filters[:, :, :, j]))

    return conv_result

def MX_C(X, f):
    n_img = X.shape[-1]
    n_patch = (32//f) * (32 // f)
    MX = np.zeros((n_patch, f * f * 3, n_img))

    for i in range(n_img):
        idx = 0
        for row in range(32 // f):
            for col in range(32 // f):
                patch = X[row * f:(row + 1) * f, col * f:(col + 1) * f, :, i]
                MX[idx, :, i] = patch.reshape(-1)
                idx += 1
    # print("MX size is ", MX.shape)
    return MX

def conv_MX(MX, filters):
    f_filter = int(np.sqrt(filters.shape[0] / 3))
    nf = filters.shape[-1]
    n_MX = MX.shape[-1]
    n_patch = (32 // f_filter) * (32 // f_filter)
    n_patch_own = MX.shape[0]
    conv_out_MX = np.zeros((n_patch_own, nf, n_MX))
    conv_out_MX = np.einsum('ijn, jl -> iln', MX, filters, optimize=True)
    conv_output_MX_flat = conv_out_MX.reshape((n_patch_own * nf, n_MX), order='C')
    return conv_out_MX

# Exercise 2
def forward_pass(MX, init_params):
    filters = init_params['Fs_flat']
    b0 = init_params['b_f']
    W1 = init_params['W1']
    b1 = init_params['b1']
    W2 = init_params['W2']
    b2 = init_params['b2']

    n_patch_own = MX.shape[0]
    n_img = MX.shape[-1]
    conv_out_MX = conv_MX(MX, filters) + b0  # First layer

    conv_out_MX_Relu = np.maximum(0, conv_out_MX) # Relu

    h = conv_out_MX_Relu.reshape((n_patch_own * filters.shape[-1], n_img), order='C')


    x1 = np.dot(W1, h) + b1  # the second layer
    x1_relu = np.maximum(0, x1)


    s = np.dot(W2, x1_relu) + b2 # the third layer

    exp_s = np.exp(s)

    sum_exp_s = np.sum(exp_s, axis=0, keepdims=True)


    p = np.exp(s) / np.sum(np.exp(s), axis=0, keepdims=True)


    return p,h,x1_relu

def back_pass(MX, Y, init_params, h, x1, p, n_patch, n_f, lam):
    filters = init_params['Fs_flat']

    f_filter = int(np.sqrt(filters.shape[0] / 3))
    nf = filters.shape[-1]
    # Fs_flat = filters.reshape((f_filter * f_filter * 3, nf), order='C')
    b0 = init_params['b_f']
    W1 = init_params['W1']
    b1 = init_params['b1']
    W2 = init_params['W2']
    b2 = init_params['b2']
    N = Y.shape[-1]
    patch = n_patch
    nf = n_f
    # print(f"patch: {patch}, nf: {nf}")
    n_MX = h.shape[-1]

    #-----------------label smoothing---------------------------------------
    Y_smooth = label_smoothing(Y,0.1)
    #-----------------------------------------------------------------------

    #---------------- Softmax + Cross Entropy ----------------------
    grad_S = p - Y_smooth
    # print("反向传播中损失对他的梯度：", grad_S.shape)

    #---------------- second layer W1,b1----------------------
    grad_W2 = (1/N) * (grad_S @ x1.T) + 2 * lam * W2 # 10 * 50
    grad_b2 = np.mean(grad_S, axis = 1, keepdims=True) # 10 * 1
    grad_x1 = W2.T @ grad_S   # 50 * N   # 传到上一层的误差

    # ---------------- first layer w1 b1----------------------
    # Relu mask
    # grad_x1_relu = grad_x1 * (x1 > 0)
    grad_x1[x1 <= 0] = 0
    grad_W1 = (1/N) * (grad_x1 @ h.T) + 2 * lam * W1
    grad_b1 = np.mean(grad_x1, axis = 1, keepdims=True)

    G = W1.T @ grad_x1 # (128, N) # 回传到卷积输出的误差


    # ---------------- 卷积层的损失梯度----------------------
    G[h <= 0] = 0
    GG = G.reshape((n_patch, n_f, N), order='C') # (64, 2, 5) # 重新展开为三维

    MXt = np.transpose(MX, (1, 0, 2))
    grad_filters = (1 / N) * np.einsum('ijn,jln->il', MXt, GG, optimize=True)  + 2 * lam * filters.reshape((f_filter * f_filter * 3, nf)) # (48, 2)

    grad_bf = np.mean(GG, axis = 2, keepdims=False).sum(axis=0).reshape(-1, 1)


    grads = {
        'W1': grad_W1,
        'b1': grad_b1,
        'W2': grad_W2,
        'b2': grad_b2,
        'filters': grad_filters,
        'bf': grad_bf
    }

    return grads

def He_initialization(shape):
    return np.random.randn(*shape).astype(np.float64) * np.sqrt(2 / shape[1])

def initialize_parameters(f_filter, nf, nh, n_p):
    np.random.seed(42)
    patches = nf * (32 // f_filter) * (32 // f_filter)
    Fs_flat = np.random.randn(f_filter * f_filter * 3 , nf) * np.sqrt(2 / (3 * f_filter * f_filter))
    b_f = np.zeros((nf, 1))
    W1 = np.random.randn(nh, patches) * np.sqrt(2 / patches)
    b1 = np.zeros((nh, 1))
    W2 = np.random.randn(10, nh) * np.sqrt(2 / nh)
    b2 = np.zeros((10, 1))


    net_params = {
        'Fs_flat': Fs_flat,
        'W1': W1,
        'W2': W2,
        'b1': b1,
        'b2': b2,
        'b_f': b_f
    }
    return net_params

def compare_grads(grad1, grad2, name):
    diff = np.abs(grad1 - grad2)
    max_diff = np.max(diff)
    print(f"{name} max diff: {max_diff:.15e}")

#Exsercise3
def seperator(filename):
    cifar_dir = r'C:\Users\JKX\Desktop\DL\Assignment3\dataset'
    with open(cifar_dir + filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X = dict[b'data'].astype(np.float64) / 255.0
    y = dict[b'labels']
    X = X.reshape(-1, 3, 32, 32)
    X = np.transpose(X, (2, 3, 1, 0))  # (32,32,3,n)
    Y = np.eye(10)[y]
    return X, Y.T, y

def load_traindata():
    X_list = []
    Y_list = []
    y_list = []

    for i in range(1,6):
        file_name = f'/data_batch_{i}'
        X_batch, Y_batch, y_batch = seperator(file_name)
        X_list.append(X_batch)
        Y_list.append(Y_batch)
        y_list.append(y_batch)

    X_all = np.concatenate(X_list, axis=3)  # (32, 32, 3, 50000)
    Y_all = np.concatenate(Y_list, axis=1)  # (10, 50000)
    y_all = np.concatenate(y_list, axis=0)  # (50000,)

    X_train = X_all[:, :, :, :49000]
    Y_train = Y_all[:, :49000]
    y_train = y_all[:49000]

    X_val = X_all[:, :, :, 49000:]
    Y_val = Y_all[:, 49000:]
    y_val = y_all[49000:]
    return X_train, Y_train, y_train, X_val, Y_val, y_val

def loss(P,Y):
    n = P.shape[1]
    loss = -np.sum(Y * np.log(P)) / n

    return loss

def ComputeAccuracy(P,y):
    P_argmax = np.argmax(P, axis=0)
    # print("P",P_argmax)
    y_argmax = np.argmax(y, axis=0)
    # print("Y",y_argmax)
    compute_accuracy = np.equal(P_argmax, y_argmax).sum() / y.shape[1]
    return compute_accuracy

def MiniBatch_GD(X_ims, Y, MX, GDparams, init_net, Y_val, MX_val):
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    lr_list = []

    net_params = copy.deepcopy(init_net)

    n = X_ims.shape[-1]
    n_epoch = GDparams['epoch']
    min_eta = GDparams['eta_min']
    max_eta = GDparams['eta_max']
    n_batch = GDparams['n_batch']
    lam = GDparams['lam']
    t = 0

    current_n_s = 800
    total_cycles = 3
    completed_cycles = 0
    t_cycle_start = 0
    current_lr = GDparams['eta_max']

    for i in tqdm.tqdm(range(n_epoch), desc="Epoch Running",  dynamic_ncols=True):
        indx = np.random.permutation(n)
        MX_train = MX[:, :, indx]
        Y_train = Y[:, indx]

        for j in tqdm.tqdm(range(n // n_batch), desc="Batch Running",  dynamic_ncols=True):
            j_start = j * n_batch
            j_end = (j + 1) * n_batch

            Xbatch = X_ims[:, :, :, indx[j_start:j_end]]

            tx = np.random.randint(-3, 4)
            ty = np.random.randint(-3, 4)
            Xbatch_aug = data_augmentation(tx, ty, Xbatch)

            MX_batch = MX_C(Xbatch_aug, f)

            Y_batch = Y_train[:, j_start:j_end]

            # forward and backward
            p_batch, h_batch,x1_batch = forward_pass(MX_batch, net_params)
            grads = back_pass(MX_batch, Y_batch, net_params, h_batch, x1_batch, p_batch,36, 60, lam)

            # CLR learning rate
            cycle = int(np.floor(t / (2 * current_n_s)))
            t_mod = t - 2 *  current_n_s * cycle
            if t_mod <=  current_n_s:
                eta_t = min_eta + (t_mod /  current_n_s) * (current_lr - min_eta)
            else:
                eta_t = current_lr - ((t_mod -  current_n_s) /  current_n_s) * (current_lr - min_eta)
            lr_list.append(eta_t)

            # update
            net_params['W1'] -= eta_t * grads['W1']
            net_params['b1'] -= eta_t * grads['b1']
            net_params['W2'] -= eta_t * grads['W2']
            net_params['b2'] -= eta_t * grads['b2']
            net_params['Fs_flat'] -= eta_t * grads['filters']
            net_params['b_f'] -= eta_t * grads['bf']

            if t_mod == 2 * current_n_s - 1:
                if cycle < total_cycles - 1:
                    print(f"Cycle {cycle} completed. Doubling steps to {current_n_s * 2}")
                    current_n_s *= 2

            t += 1

            P_train,h_train,x1_relu_train = forward_pass(MX_batch,  net_params)
            train_loss = loss(P_train, Y_batch)
            train_acc = ComputeAccuracy(P_train, Y_batch)

            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)

            # --- Evaluate on whole val set ---
            P_val,h_val,x1_relu_val = forward_pass(MX_val, net_params)
            loss_test = loss(P_val, Y_val)
            val_loss_list.append(loss_test)
            val_acc = ComputeAccuracy(P_val, Y_val)
            val_acc_list.append(val_acc)
        current_lr /= 2

    return train_loss_list, train_acc_list, val_loss_list, val_acc_list, lr_list, net_params

def label_smoothing(Y,epsilon=0.1):
    K = Y.shape[0]
    Y_smoothing = (1 - epsilon) * Y + epsilon / (K - 1) * (1 - Y)
    return Y_smoothing

def data_augmentation(tx, ty, X, p=0.5):
    X_aug = np.copy(X)
    H, W, C, N = X.shape
    for i in range(N):
        if np.random.rand() < p:
            img = X[:, :, :, i]

            img = np.fliplr(img)

            img_shifted = np.zeros_like(img)

            if ty >= 0:
                img_shifted[ty:, :, :] = img[:H - ty, :, :]
            else:
                img_shifted[:H + ty, :, :] = img[-ty:, :, :]

            temp = np.zeros_like(img)
            if tx >= 0:
                temp[:, tx:, :] = img_shifted[:, :W - tx, :]
            else:
                temp[:, :W + tx, :] = img_shifted[:, -tx:, :]

            X_aug[:, :, :, i] = temp

    return X_aug

if __name__ == '__main__':
    batch1_name = '\data_batch_1'
    batch2_name = '\data_batch_2'
    batch3_name = '\data_batch_3'
    batch4_name = '\data_batch_4'
    batch5_name = '\data_batch_5'
    test_name = r'\test_batch'
    batch1_train, batch1_label_onehot, batch1_label_true = seperator(batch1_name)
    batch2_train, batch2_label_onehot, batch2_label_true = seperator(batch2_name)
    batch3_train, batch3_label_onehot, batch3_label_true = seperator(batch3_name)
    batch4_train, batch4_label_onehot, batch4_label_true = seperator(batch4_name)
    batch5_train, batch5_label_onehot, batch5_label_true = seperator(batch5_name)
    test1_train, test1_label_onehot, test1_label_true = seperator(test_name)
    train_X, train_Y, train_y, val_X, val_Y, val_y = load_traindata() # get the train data 32 * 32 * 3 * 49000


    # mean_train = np.mean(train_X, axis=(0, 1, 2))
    # std_train = np.std(train_X, axis=(0, 1, 2))
    mean_train = np.mean(train_X, axis=(0, 1, 3), keepdims=True)  # 形状变成 (1,1,3,1)
    std_train = np.std(train_X, axis=(0, 1, 3), keepdims=True)
    train_X_nor = (train_X - mean_train) / std_train

    # mean_val = np.mean(val_X, axis=(0, 1, 3), keepdims=True)
    # std_val = np.std(val_X, axis=(0, 1, 3), keepdims=True)
    val_X_nor = (val_X - mean_train) / std_train

    # mean_test = np.mean(test1_train, axis=(0, 1, 2))
    # std_test = np.std(test1_train, axis=(0, 1, 2))
    test_X_nor = (test1_train - mean_train) / std_train


    f, nf, nh = 5, 60, 400
    GDparams = {
        'eta_min': 1e-5,
        'eta_max': 5e-2,
        'n_batch': 50,
        'epoch': 75,
        'lam':0.001,
        'n_s':800
    }

    MX_train = MX_C(train_X_nor, f)
    MX_val = MX_C(val_X_nor, f)
    init_params = initialize_parameters(f, nf, nh, 64)
    train_loss_list, train_acc_list, val_loss_list, val_acc_list, lr_list, trained_params = MiniBatch_GD(
        train_X_nor, train_Y, MX_train, GDparams, init_params, val_Y, MX_val)
    MX_test = MX_C(test_X_nor, f)
    P_test, h_test, x1_relu_test = forward_pass(MX_test, trained_params)
    test_accuracy = ComputeAccuracy(P_test, test1_label_onehot)
    print("The final accuracy in bath is", test_accuracy)

    plt.figure()
    plt.plot(train_loss_list, color = 'red',label='Train loss')
    plt.plot(val_loss_list, color = 'blue',label='Test loss')
    plt.xlabel('Update step')
    plt.ylabel('Loss')
    plt.legend(loc = 'upper right')
    plt.show()

    plt.figure()
    plt.plot(train_acc_list, color = 'red',label='Train acc')
    plt.plot(val_acc_list, color = 'blue',label='Test acc')
    plt.xlabel('Update step')
    plt.ylabel('Accuracy')
    plt.legend(loc = 'upper right')
    plt.show()

    plt.figure()
    plt.plot(lr_list, color = 'red',label='Learning rate')
    plt.xlabel('Update step')
    plt.ylabel('Learning rate')
    plt.legend(loc = 'upper right')
    plt.show()
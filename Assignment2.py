import copy
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tqdm
from torch_gradient_computations import ComputeGradsWithTorch
import time

# Read in the data and initialize the parameters of the network
def seperator(filename):
    cifar_dir = r'C:\Users\JKX\Desktop\DL\Assignment1\dataset'
    with open(cifar_dir + filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X = dict[b'data'].astype(np.float64) / 255.0
    y = dict[b'labels']
    Y = np.eye(10)[y]
    return X.T, Y.T, y                                        # return train_data, one-hot label and true label

def normalize(mean_X_train, std_X_train, target_data):
    after_norm = (target_data - mean_X_train) / std_X_train
    return after_norm

def initial_net(L, d, m):                                     # need the number of layers, dimension and size of layer
    rng = np.random.default_rng(seed = 42)
    BitGen = type(rng.bit_generator)
    seed = 42
    rng.bit_generator.state = BitGen(seed).state
    net_params = {}
    net_params['W'] = [None] * L
    net_params['b'] = [None] * L

    net_params['W'][0] = rng.standard_normal(size = (50 , 3072)) / np.sqrt(d)
    net_params['W'][1] = rng.standard_normal(size = (10 , 50)) / np.sqrt(m)

    net_params['b'][0] = np.zeros((50 , 1))
    net_params['b'][1] = np.zeros((10 , 1))

    return net_params

def ApplyNetwork(X, net_params):
    fp_data = {}
    s_1 = net_params['W'][0].dot(X) + net_params['b'][0]
    fp_data['s_1'] = s_1
    h = np.maximum(0, s_1)                                   # ReLu
    fp_data['h'] = h
    s_2 = net_params['W'][1].dot(h) + net_params['b'][1]
    fp_data['s_2'] = s_2
    exp_s = np.exp(s_2)
    sum_exp = np.sum(exp_s, axis=0, keepdims=True)
    P = exp_s / sum_exp
    return P, fp_data

def Backwardpass(X, Y, P, net_params, fp_data, lam):
    grads = {}
    grads['W'] = [None] * 2
    grads['b'] = [None] * 2
    N = X.shape[1]
    dS = -(Y - P)
    grads['W'][1] = (1 / N) *np.dot(dS, fp_data['h'].T) + 2 * lam * net_params['W'][1]
    grads['b'][1] = (1 / N) *np.sum(dS, axis=1, keepdims=True)

    G_hidded = np.dot(net_params['W'][1].T, dS)
    G_hidded[fp_data['h'] <= 0] = 0

    grads['W'][0] = (1 / N) *np.dot(G_hidded, X.T) + 2 * lam * net_params['W'][0]
    grads['b'][0] = (1 / N) *np.sum(G_hidded, axis=1, keepdims=True)

    return grads

def relative_error(my_grads, torch_grads, eps=1e-10):
    max_rel_error = 0

    for i in range(len(my_grads['W'])):
        num = np.abs(my_grads['W'][i] - torch_grads['W'][i])
        denom = np.maximum(eps, np.abs(my_grads['W'][i]) + np.abs(torch_grads['W'][i]))
        rel_error_W = np.max(num / denom)

        num_b = np.abs(my_grads['b'][i] - torch_grads['b'][i])
        denom_b = np.maximum(eps, np.abs(my_grads['b'][i]) + np.abs(torch_grads['b'][i]))
        rel_error_b = np.max(num_b / denom_b)
        
        max_rel_error = max(max_rel_error, rel_error_W, rel_error_b)

        print(f"Layer {i} - rel_error_W: {rel_error_W:.2e}, rel_error_b: {rel_error_b:.2e}")

    return max_rel_error

def MiniBatchGD(X, Y, GDparams, init_net, lam, X_val, Y_val):
    train_loss_list = []
    train_cost_list = []
    train_accuracy_list = []
    val_loss_list = []
    val_cost_list = []
    val_accuracy_list = []
    lr_list = []
    trained_net = copy.deepcopy(init_net)
    n = X.shape[1]
    t = 0
    for i in tqdm.tqdm(range(GDparams['n_epochs']), desc="Epoch Running"):
        random = np.random.permutation(n)

        for j in range(n // GDparams['n_batch']):
            j_start = j * GDparams['n_batch']
            j_end = (j + 1) * GDparams['n_batch']
            inds = random[j_start:j_end]
            Xbatch = X[:, inds]
            Ybatch = Y[:, inds]

           # Exercise3
            cycle = int(np.floor(t / (2 * GDparams['n_s'])))
            t_mod = t - 2 * GDparams['n_s'] * cycle
            if t_mod <=GDparams['n_s']:
                eta_t = GDparams['eta_min'] + (t_mod / GDparams['n_s']) * (GDparams['eta_max'] - GDparams['eta_min'])
            else:
                eta_t = GDparams['eta_max'] - ((t_mod - GDparams['n_s']) / GDparams['n_s']) * (GDparams['eta_max'] - GDparams['eta_min'])
            lr_list.append(eta_t)
            P_1,fp_data_1 = ApplyNetwork(Xbatch, trained_net)
            grads_my = Backwardpass(Xbatch, Ybatch, P_1, trained_net, fp_data_1, lam)
            for k in range(len(trained_net['W'])):
                trained_net['W'][k] -= eta_t * grads_my['W'][k]
                trained_net['b'][k] -= eta_t * grads_my['b'][k]
            t += 1
            P_2, fp_data_2 = ApplyNetwork(X, trained_net)
            cost_train, loss_train = cost_function(P_2, Y, X, lam, trained_net)
            train_accuracy = ComputeAccuracy(P_2, Y)
            train_loss_list.append(loss_train)
            train_cost_list.append(cost_train)
            train_accuracy_list.append(train_accuracy)

        P_3, fp_data_3 = ApplyNetwork(X_val, trained_net)
        cost_val, loss_val = cost_function(P_3, Y_val, X_val, lam, trained_net)
        val_loss_list.append(loss_val)
        val_cost_list.append(cost_val)
        val_accuracy = ComputeAccuracy(P_3, Y_val)
        val_accuracy_list.append(val_accuracy)

    return trained_net ,train_loss_list, train_cost_list, train_accuracy_list, val_loss_list, val_cost_list, val_accuracy, lr_list


def loss_function(P,y):
    loss = -y * np.log(P)
    return loss

def cost_function(P, y,x, lam, network):
    loss = (1 / x.shape[1]) * np.sum(loss_function(P,y))
    sun_weight = sum(np.sum(W**2) for W in network['W'])
    cost = loss + lam * sun_weight
    return cost, loss

def ComputeAccuracy(P,y):
    P_argmax = np.argmax(P, axis=0)
    # print("P",P_argmax)
    y_argmax = np.argmax(y, axis=0)
    # print("Y",y_argmax)
    compute_accuracy = np.equal(P_argmax, y_argmax).sum() / y.shape[1]
    return compute_accuracy

# Exercise 4
def coarse_search_lam(X_train, Y_train, X_val, Y_val, lam_min, lam_max, num_lam):
    rng = np.random.default_rng()
    l = [lam_min + (lam_max - lam_min) * rng.random() for _ in range(num_lam)]
    lam_candidates = 10 ** np.array(l)

    GDparams = {}
    GDparams['n_batch'] = 100
    GDparams['eta_min'] = 1e-5
    GDparams['eta_max'] = 1e-1
    result = []
    n = X_train.shape[1]
    GDparams['n_s'] = 2 *np.floor(n / GDparams['n_batch'])
    update_step = 4 * GDparams['n_s']
    GDparams['n_epochs'] =int(update_step / (n // GDparams['n_batch']))
    for lam in tqdm.tqdm(lam_candidates, desc="Searching lambda candidates") :
        init_net = initial_net(L=2, d=3072, m=50)
        trained_net,train_loss_list, train_cost_list, train_accuracy_list, val_loss_list, val_cost_list, val_accuracy_list, lr_list = MiniBatchGD(X_train, Y_train, GDparams, init_net, lam, X_val, Y_val)
        final_accuracy = np.max(val_accuracy_list)
        result.append({'lam':lam,
                       'val_acc':final_accuracy,
                       'GDparams':copy.deepcopy(GDparams),
                       'trained_net': trained_net,
                       'train_loss_list':train_loss_list,
                       'train_cost_list':train_cost_list,
                       'train_accuracy_list':train_accuracy_list,
                       'val_loss_list':val_loss_list,
                       'val_cost_list':val_cost_list,
                       'val_accuracy_list':val_accuracy_list,
                       'lr_list':lr_list})

    sorted_results = sorted(result, key=lambda x: x['val_acc'], reverse=True)
    best_lambda = sorted_results[0]['lam']
    best_acc = sorted_results[0]['val_acc']
    top3 = sorted_results[:3]

    plt.figure()
    for i, r in enumerate(top3, 1):
        plt.plot(r['train_loss_list'], label=f"Top {i}: lam={r['lam']:.5e}")
    plt.xlabel("Update Step")
    plt.ylabel("Train Loss")
    plt.legend(loc='upper right')
    plt.title("Train Loss Curves for Top 3 Candidates")
    plt.show()

    plt.figure()
    for i, r in enumerate(top3, 1):
        plt.plot(r['val_loss_list'], label=f"Top {i}: lam={r['lam']:.5e}")
    plt.xlabel("Epochs ")
    plt.ylabel("Validation Loss")
    plt.legend(loc='upper right')
    plt.title("Validation Loss Curves for Top 3 Candidates")
    plt.show()

    # final_accuracies = [r['val_acc'] for r in top3]
    # lam_values = [r['lam'] for r in top3]
    #
    # plt.figure()
    # plt.bar(range(len(final_accuracies)), final_accuracies, tick_label=[f"lam={lam:.5e}" for lam in lam_values])
    # plt.xlabel("Candidate")
    # plt.ylabel("Final Accuracy")
    # plt.legend(loc='upper right')
    # plt.title("Final Accuracy of Top 3 Candidates")
    # plt.show()

    # plt.figure()
    # for i, r in enumerate(top3, start=1):
    #     plt.plot(r['val_accuracy_list'], label=f"Candidate {i}: lam={r['lam']:.5e}, final_acc={r['val_acc']:.5f}")
    # plt.xlabel("Epochs or Update Step")
    # plt.ylabel("Validation Accuracy")
    # plt.title("Validation Accuracy Curves of Top 3 Candidates")
    # plt.legend(loc="upper right")
    # plt.title("Validation Accuracy Curves of Top 3 Candidates")
    # plt.show()

    plt.figure()
    plt.plot(sorted_results[0]['lr_list'],color = "Red", label="Learning Rate")
    plt.xlabel("Epochs or Update Step")
    plt.ylabel("Learning Rate")
    plt.legend(loc='upper right')
    plt.title("Learning Rate changed")
    plt.show()

    # best_entry = max(result, key=lambda x: x['val_acc'])
    # best_lambda = best_entry['lam']
    # highest_acc = best_entry['val_acc']

    lam_min_val = 10 ** lam_min
    lam_max_val = 10 ** lam_max
    print(">>> (i) Range of the values searched for lam:")
    print(f"    lam_min = 10^{lam_min} = {lam_min_val:.2e}")
    print(f"    lam_max = 10^{lam_max} = {lam_max_val:.2e}\n")

    cycles = int(update_step / (2 * GDparams['n_s']))
    print(">>> (ii) Number of cycles used for training during the coarse search:")
    print(f"    We used {cycles} cycle(s) in total.\n")

    print(">>> (iii) Hyper-parameter settings for the 3 best performing networks:")
    for i, r in enumerate(top3, start=1):
        print(f"  - Network {i}:")
        print(f"      lam      = {r['lam']:.6e}")
        print(f"      val_acc  = {r['val_acc']:.4f}")
        print(f"      n_batch  = {r['GDparams']['n_batch']}")
        print(f"      eta_min  = {r['GDparams']['eta_min']}")
        print(f"      eta_max  = {r['GDparams']['eta_max']}")
        print(f"      n_s      = {r['GDparams']['n_s']}")
        print(f"      n_epochs = {r['GDparams']['n_epochs']}")
        print("")

    print("The highest accuracy on train is:", np.max(sorted_results[0]['train_accuracy_list']))
    print("Best lambda:", best_lambda, "Highest val accuracy:", best_acc)
    return best_lambda, best_acc, sorted_results, sorted_results[0]['trained_net']
if __name__ == '__main__':
    # filename_train = '\data_batch_1'
    # filename_val = '\data_batch_2'
    # filename_test = r'\test_batch'
    #
    # X_train, Y_train, y_train = seperator(filename_train)
    # X_val, Y_val, y_val = seperator(filename_val)
    # X_test, Y_test, y_test = seperator(filename_test)
    #
    # mean_X_train = np.mean(X_train, axis=1).reshape(3072, 1)
    # std_X_train = np.std(X_train, axis=1).reshape(3072, 1)
    # X_train_after = normalize(mean_X_train, std_X_train, X_train)
    # X_val_after = normalize(mean_X_train, std_X_train, X_val)
    # X_test_after = normalize(mean_X_train, std_X_train, X_test)

    # gradient consumption check
    # d_small = 5
    # n_small = 3
    # m = 6
    # lam = 0
    #
    # small_net = {}
    # L = 2
    # small_net['W'] = [None] * L
    # small_net['b'] = [None] * L
    # rng = np.random.default_rng()
    # small_net['W'][0] = (1/np.sqrt(d_small))*rng.standard_normal(size = (m, d_small))
    # small_net['b'][0] = np.zeros((m, 1))
    # small_net['W'][1] = (1 / np.sqrt(m)) * rng.standard_normal(size=(10, m))
    # small_net['b'][1] = np.zeros((10, 1))
    #
    # X_small = X_train_after[0:d_small, 0:n_small]
    # Y_small = Y_train[:, 0:n_small]
    # print("X_small", X_small.shape)
    # print("Y_small", Y_small.shape)
    # P_small, fp_data = ApplyNetwork(X_small, small_net)
    # my_grads = Backwardpass(X_small, Y_small, P_small,small_net, fp_data, lam)
    # torch_grads = ComputeGradsWithTorch(X_small, y_train[0:n_small], small_net)
    #
    #
    # A = relative_error(my_grads, torch_grads)
    # print("The relative error：", A)

    # use 100 samples to check
    # GDparams = {}
    # GDparams['n_epochs'] = 200
    # GDparams['n_batch'] = 10
    # GDparams['eta'] = 0.01
    # lam = 0.01
    # train_loss_lits = []
    # train_cost_lits = []
    # train_accuracy_list = []
    # init_net = initial_net(2, 3072, 50)
    # trained_net, train_loss_list, train_cost_list, train_accuracy_list = MiniBatchGD(X_train_after[:, :100], Y_train[:, :100], GDparams, init_net , lam)
    # print("Accuracy:", train_accuracy_list)
    # plt.plot(train_loss_list, color="Green", label="Training Loss")
    # plt.legend(loc='upper right')
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.show()

    # Exercise 3
    # GDparams = {}
    # GDparams['n_epochs'] = 48
    # GDparams['n_batch'] = 100
    # GDparams['eta_min'] = 1e-5
    # GDparams['eta_max'] = 1e-1
    # GDparams['n_s'] = 800
    # lam = 0.01
    # train_loss_lits = []
    # train_cost_lits = []
    # train_accuracy_lits = []
    # val_loss_list = []
    # val_cost_list = []
    # val_accuracy_list = []
    # lr_list = []
    # init_net = initial_net(2, 3072, 50)
    # trained_net, train_loss_list, train_cost_list, train_accuracy_list, val_loss_list, val_cost_list, val_accuracy_list, lr_list = MiniBatchGD(X_train_after,Y_train, GDparams,init_net, lam ,X_val_after, Y_val)
    #
    # plt.figure()
    # plt.plot(train_loss_list, color = "Green", label='train loss')
    # plt.plot(val_loss_list, color = "Red", label='val loss')
    # plt.legend(loc='upper right')
    # plt.xlabel("Update Step")
    # plt.ylabel("Loss")
    # plt.show()
    #
    # plt.figure()
    # plt.plot(train_cost_list, color = "Green", label='train cost')
    # plt.plot(val_cost_list, color = "Red", label='val cost')
    # plt.legend(loc='upper right')
    # plt.xlabel("Update Step")
    # plt.ylabel("Cost")
    # plt.show()
    #
    # plt.figure()
    # plt.plot(train_accuracy_list, color = "Green", label='train accuracy')
    # plt.plot(val_accuracy_list, color = "Red", label='val accuracy')
    # plt.legend(loc='upper right')
    # plt.xlabel("Update Step")
    # plt.ylabel("Accuracy")
    # plt.show()
    #
    # plt.figure()
    # plt.plot(lr_list, color = "Purple", label='Learning Rate')
    # plt.legend(loc='upper right')
    # plt.xlabel("Update Step")
    # plt.ylabel("Learning Rate")
    # plt.show()

    # Exercise4
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
    all_data_train = np.hstack((batch1_train, batch2_train, batch3_train, batch4_train, batch5_train))  # 3072 * 50000
    all_data_label_onehot = np.hstack((batch1_label_onehot, batch2_label_onehot, batch3_label_onehot, batch4_label_onehot, batch5_label_onehot))  # 10 * 50000

    train_data = all_data_train[:, :49000]  # 3072 * 45000
    val_data = all_data_train[:, 49000:]  # 3072 * 5000
    train_onehot = all_data_label_onehot[:, :49000]  # 10 * 45000
    val_onehot = all_data_label_onehot[:, 49000:]  # 10 * 5000

    mean_train_data = np.mean(train_data, axis=1).reshape(3072,1)  # the mean of the training data         3072 * 1
    std_train_data = np.std(train_data, axis=1).reshape(3072,1)  # the standard of the training data    3072 * 1
    mean_val_data = np.mean(val_data, axis=1).reshape(3072,1)
    std_val_data = np.std(val_data, axis=1).reshape(3072,1)
    mean_test_data = np.mean(test1_train, axis=1).reshape(3072,1)
    std_test_data = np.std(test1_train, axis=1).reshape(3072,1)

    train_data_after = normalize(mean_train_data, std_train_data, train_data)   # normalize the training data            3027 * 45000
    val_data_after = normalize(mean_val_data, std_val_data, val_data)
    test_data_after = normalize(mean_test_data, std_test_data, test1_train)

    # Exercise 4
   # lam_min = -4.7131
   # lam_max = -3.2610
   # best_lam, highest_accuracy, result_list, trained_net = coarse_search_lam(train_data_after, train_onehot,val_data_after, val_onehot,lam_min, lam_max, 8 )

    # On the test data
    GDparams = {}
    GDparams['n_batch'] = 100
    GDparams['eta_min'] = 1e-5
    GDparams['eta_max'] = 1e-1
    n = train_data_after.shape[1]
    GDparams['n_s'] = 2 * np.floor(n / GDparams['n_batch'])
    update_step = 6 * GDparams['n_s']
    GDparams['n_epochs'] = int(update_step / (n // GDparams['n_batch']))
    init_net = initial_net(L=2, d=3072, m=50)
    lam = 6.784270700693171e-05
    trained_net, train_loss_list, train_cost_list, train_accuracy_list, val_loss_list, val_cost_list, val_accuracy_list, lr_list = MiniBatchGD(train_data_after,
                                                                                                                                               train_onehot, GDparams,
                                                                                                                                               init_net, lam, val_data_after, val_onehot)
    P_test, fapa_data = ApplyNetwork(test_data_after, trained_net)
    Accuracy = ComputeAccuracy(P_test, test1_label_onehot)

    plt.figure()
    plt.plot(train_loss_list, color='blue', label='Train Loss')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

    plt.figure()
    plt.plot(val_loss_list, color='purple', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()
    print("Accuracy on test set:", Accuracy)



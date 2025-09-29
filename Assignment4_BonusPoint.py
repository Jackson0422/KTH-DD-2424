import numpy as np
import matplotlib.pyplot as plt
import tqdm
import torch_gradient_computations_column_wise
import time

# Exercise 0.3
def generate_sequence(h0, x0, n, RNN_params, int_to_char, T, p=0.9):
    """
    :param h0: the hidden state at time 0
    :param x0: first input vector to RNN
    :param n: length of sequence
    :param RNN_params: RNN parameters
    :return: one-hot K * n,
    """
    W = RNN_params['W']
    U = RNN_params['U']
    b = RNN_params['b']
    V = RNN_params['V']
    c = RNN_params['c']

    h_t = h0
    x_t = x0
    K = U.shape[1]

    Y = np.zeros((K, n), dtype = int)
    generate_text = []

    for i in range(n):
        a = W @ h_t + U @ x_t + b
        h_t = np.tanh(a)
        o = V @ h_t + c

        o /= T
        exp_s = np.exp(o)
        sum_exp_s = np.sum(exp_s, axis=0, keepdims=True)
        p_t = exp_s / sum_exp_s

        sorted_indices = np.argsort(-p_t, axis=0).flatten()
        sorted_probs = p_t[sorted_indices]

        cumulative_probs = np.cumsum(sorted_probs)
        k_t = np.searchsorted(cumulative_probs, p) + 1

        top_p_indices = sorted_indices[:k_t]
        top_p_probs = p_t[top_p_indices]
        top_p_probs /= np.sum(top_p_probs)

        sampled_index = np.random.choice(top_p_indices, p=top_p_probs.flatten())
        Y[sampled_index, i] = 1
        generate_text.append(int_to_char[sampled_index])

        x_t = np.zeros((K, 1))
        x_t[sampled_index] = 1

        # cp = np.cumsum(p_t, axis=0)
        # a = rng.uniform(size = 1)
        # ii = np.argmax(cp - a > 0)
        #
        # Y[ii, i] = 1
        # generate_text.append(int_to_char[ii])
        #
        # x_t = np.zeros((K, 1))
        # x_t[ii] = 1

    return Y, "".join(generate_text)

def to_one_hot(chars, char_to_ind, K):
    """
    :param chars:sequence of cahr
    :param char_to_ind: from char to int
    :param K:the number of class
    :return:
    """

    seq_length = len(chars)
    X = np.zeros((K, seq_length), dtype = int)

    for t, ch in enumerate(chars):
        index = char_to_ind[ch]
        X[index, t] = 1

    return X

# Exercise 0.4
def forward_pass(X, Y, h0, RNN_params):
    """
    :param X: inout data
    :param Y:  one-hot
    :param h0: begin hidden state
    :param RNN_params: RNN parameters
    :return: loss, h, a, o, p
    """

    W = RNN_params['W']
    U = RNN_params['U']
    b = RNN_params['b']
    V = RNN_params['V']
    c = RNN_params['c']

    seq_len = X.shape[1]
    m = h0.shape[0]
    K = Y.shape[0]

    h = np.zeros((m, seq_len))
    a = np.zeros((m, seq_len))
    o = np.zeros((K, seq_len))
    p = np.zeros((K, seq_len))

    non_zero_indices = np.argmax(X, axis=0)
    precomputed_UX = U[:, non_zero_indices]

    h_begin = h0
    loss = 0

    for t in range(seq_len):
        x_t = X[:, t:t+1]

        a_t = W @ h_begin + precomputed_UX[:, t:t+1] + b
        h_t = np.tanh(a_t)
        o_t = V @ h_t + c
        o_t -= np.max(o_t)
        exp_s = np.exp(o_t)
        sum_exp_s = np.sum(exp_s, axis=0, keepdims=True)
        p_t = exp_s / sum_exp_s

        y_t = Y[:, t:t+1]
        loss_t = -np.sum(y_t * np.log(p_t))
        loss += loss_t

        h[:, t:t+1] = h_t
        a[:, t:t+1] = a_t
        o[:, t:t+1] = o_t
        p[:, t:t+1] = p_t

        h_begin = h_t

    loss = loss / seq_len

    return loss, h, a, o, p

def backward_pass(X, Y, h, a, p, RNN_params):
    """
    :param X: input data
    :param Y: label; data
    :param h: hidden state
    :param p: prediction
    :param RNN_params: net_params
    :return: grad dict
    """
    W = RNN_params['W']
    U = RNN_params['U']
    b = RNN_params['b']
    V = RNN_params['V']
    c = RNN_params['c']

    grads = {
        'W':np.zeros_like(W),
        'U':np.zeros_like(U),
        'b':np.zeros_like(b),
        'V':np.zeros_like(V),
        'c':np.zeros_like(c)
    }

    seq_length = X.shape[1]

    m = W.shape[0]
    K = V.shape[0]

    # dh_next = np.zeros((W.shape[0], 1))
    dh_next = np.zeros((m, 1))
    non_zero_indices = np.argmax(X, axis=0)
    for t in reversed(range(seq_length)):
        g_o = p[:, t:t+1] - Y[:, t:t+1]
        grads['V'] += g_o @ h[:, t : t+1].T
        grads['c'] += np.sum(g_o, axis=1, keepdims=True)

        dh = V.T @ g_o + dh_next
        da = dh * (1 - np.tanh(a[:, t:t+1]) ** 2)

        h_prev = h[:, t - 1:t] if t > 0 else np.zeros_like(h[:, 0:1])
        grads['b'] += np.sum(da, axis=1, keepdims=True)
        grads['W'] += np.outer(da.flatten(), h_prev.flatten())
        grads['U'] += da @ X[:, t:t+1].T

        dh_next = W.T @ da

    for param in grads:
        grads[param] /= seq_length

    return grads

def check_grads(manual_grads, torch_grads):
    for param in manual_grads.keys():
        manual_grad = manual_grads[param]
        torch_grad = torch_grads[param]
        max_diff = np.max(np.abs(manual_grad - torch_grad))
        print(f'{param}: {max_diff}')


# Exercise 0.5
def adam_update(RNN, grads, m_adam, v_adam, t, eta = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
    """
    :param RNN:
    :param grads:
    :param m_adam: first-middle momentum
    :param v_adam: second-middle momentum
    :param t:
    :param eta:
    :param beta1:
    :param beta2:
    :param epsilon:
    :return:
    """
    for param in RNN.keys():
        m_adam[param] = beta1 * m_adam[param] + (1 - beta1) * grads[param]
        v_adam[param] = beta2 * v_adam[param] + (1 - beta2) * grads[param] ** 2

        m_hat = m_adam[param] / (1 - beta1 ** t)
        v_hat = v_adam[param] / (1 - beta2 ** t)

        RNN[param] -= eta * m_hat / (np.sqrt(v_hat) + epsilon)

def SGD(seq_length, book_data, RNN, train_param, Adam_param, L=10, batch_size = 4,n_updates = 1000000, print_every=100, sample_every=10000):
    """
    :param seq_length:
    :param book_data:
    :param RNN:
    :param train_param:
    :param Adam_param:
    :param n_updates:
    :param print_every:
    :param sample_every:
    :return:
    """
    m_adam = {param: np.zeros_like(RNN[param]) for param in RNN}
    v_adam = {param: np.zeros_like(RNN[param]) for param in RNN}
    beta1 = Adam_param['beta_1']
    beta2 = Adam_param['beta_2']
    epsilon = Adam_param['epsilon']
    eta = train_param['eta']
    M = train_param['M']
    K = train_param['K']
    smoothing_loss = -np.log(1 / K)

    train_chunks, val_chunk = split_book(book_data, L)

    t = 0
    smoothing_loss_list = []

    while t < n_updates:
        chunk = np.random.choice(train_chunks)
        chunk_length = len(chunk)

        hprev_batch = [np.zeros((M, 1)) for _ in range(batch_size)]
        batch_loss = 0
        for batch in range(batch_size):
            e = np.random.randint(0, chunk_length - seq_length - 1)

            X_chars = chunk[e:e + seq_length]
            Y_chars = chunk[e+1:e+seq_length+1]
            X = to_one_hot(X_chars, char_to_ind, K)
            Y = to_one_hot(Y_chars, char_to_ind, K)

            loss, h, a, o, p = forward_pass(X, Y, hprev_batch[batch], RNN)

            grads = backward_pass(X, Y, h, a, p, RNN)

            batch_loss += loss

            hprev_batch[batch] = h[:, -1:]

            if batch == 0:
                batch_grads = grads
            else:
                for param in grads:
                    batch_grads[param] += grads[param]

        batch_loss /= batch_size
        smoothing_loss = 0.999 * smoothing_loss + 0.001 * batch_loss
        smoothing_loss_list.append(smoothing_loss)

        for param in batch_grads:
            batch_grads[param] /= batch_size

        t += 1
        adam_update(RNN, batch_grads, m_adam, v_adam, t, eta, beta1, beta2, epsilon)

        if t % print_every == 0:
            print(f"Iteration {t}, Smooth Loss: {smoothing_loss:.4f}")

        if t == 1 or t % sample_every == 0:
            Y_sample, text_sample = generate_sequence(hprev_batch[0], X[:, 0:1], 200, RNN, int_to_char, 1, p = 0.9)
            print(f"\nSampled Text (Iteration {t}):\n{text_sample}\n")

    return smoothing_loss_list

# Exercise 2.1
def split_book(book_data, L=10):
    chunk_size = len(book_data) // L
    chunks = [book_data[i * chunk_size:(i + 1) * chunk_size] for i in range(L)]

    val_chunk = chunks[-1]
    train_chunks = chunks[:-1]

    return train_chunks, val_chunk

if __name__ == '__main__':
    start_time = time.time()
    # Exercise1 0.1
    book_dir = r'C:\Users\JKX\Desktop\DL\Assignment4'
    book_fname = book_dir + '\goblet_book.txt'
    fid = open(book_fname, 'r', encoding="utf-8")
    book_data = fid.read()          # length of book_data is 1107542
    fid.close()

    unique_chars = list(set(book_data))

    char_to_ind, int_to_char = {}, {}

    for i in range(len(unique_chars)):
        char_to_ind[unique_chars[i]] = i
        int_to_char[i] = unique_chars[i]

    # Exercise1 0.2
    m = 100 # hidden state
    eta = 0.001 # learning rate
    seq_length = 25 # length of input sequence
    K = len(unique_chars)  # output dimensionality 80
    rng = np.random.RandomState(42)
    RNN = {
        'b': np.zeros((m, 1)),
        'c': np.zeros((K, 1)),
        'U':(1/np.sqrt(2 * K)) * rng.standard_normal(size = (m, K)),
        'W':(1/np.sqrt(2 * m)) * rng.standard_normal(size = (m, m)),
        'V':(1/np.sqrt(m)) * rng.standard_normal(size = (K, m)),
    }

    # Exercise 0.4
    # X_chars = book_data[0 : seq_length]
    # Y_chars = book_data[1 : seq_length + 1]
    #
    # h0 = np.zeros((m, 1))
    #
    # m_check = 10
    # X_check = to_one_hot(X_chars, char_to_ind, K)
    # Y_check = to_one_hot(Y_chars, char_to_ind, K)
    # print("Y_check shape:", Y_check.shape)
    # print("Y_check sample:", Y_check[:, :5])
    # print("Labels (Y_check):", np.argmax(Y_check, axis=0))
    # h0_check = np.zeros((m_check, 1))
    # RNN_check = {
    #     'b': np.zeros((m_check, 1)),
    #     'c': np.zeros((K, 1)),
    #     'U': (1 / np.sqrt(2 * K)) * rng.standard_normal(size=(m_check, K)),
    #     'W': (1 / np.sqrt(2 * m_check)) * rng.standard_normal(size=(m_check, m_check)),
    #     'V': (1 / np.sqrt(m_check)) * rng.standard_normal(size=(K, m_check)),
    # }
    #
    # loss, h, a, o, p = forward_pass(X_check, Y_check, h0_check, RNN_check)
    #
    # manual_grads = backward_pass(X_check, Y_check, h, a, p, RNN_check)
    #
    # torch_grads,P_torch,loss_torch = torch_gradient_computations_column_wise.ComputeGradsWithTorch(X_check, np.argmax(Y_check, axis=0), h0_check, RNN_check)
    #
    # check_grads(manual_grads, torch_grads)
    # P_torch = P_torch.detach().cpu().numpy().astype(np.float64)
    # print("自己的P的形状为：",p.shape)
    # print("torch P的形状为",P_torch.shape)
    # max_diff = np.max(np.abs(p - P_torch))
    # print("前向传播的差值为：",max_diff)
    # loss_torch = loss_torch.detach().cpu().numpy().astype(np.float64)
    # print("Loss 差值:", np.abs(loss - loss_torch))

    # Exercise 0.5
    train_param = {
        'M':100,
        'K':80,
        'eta':0.001,
        'seq_length':seq_length,
        'n_epochs':100,
        'smoothing_loss_list': -np.log(1 / K)
    }
    Adam_param = {
        'beta_1': 0.9,
        'beta_2': 0.999,
        'epsilon': 1e-8,
    }

    smoothing_loss_list = SGD(seq_length, book_data, RNN, train_param, Adam_param, L = 20, batch_size=2, n_updates=100000, print_every=100, sample_every=10000)

    plt.figure()
    plt.plot(smoothing_loss_list, color='red', label='smoothing loss')
    plt.xlabel('Update step')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

    h0 = np.zeros((100, 1))
    x0 = np.zeros((len(char_to_ind), 1))
    x0[char_to_ind[' ']] = 1

    end_time = time.time()
    print(f"Running time is: {end_time - start_time:.4f} seconds")

    # for i in [0.9, 0.6, 0.3]:
    #     Y_1000, text_1000 = generate_sequence(h0, x0, 1000, RNN, int_to_char, 1, p = i)
    #
    #     print("\nGenerated 1000 characters text from the best model:")
    #     print(text_1000)
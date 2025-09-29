import torch
import numpy as np
def ComputeGradsWithTorch(X_ims, y, filters, network_params):
    
    X_torch = torch.from_numpy(X_ims).float()
    n = X_ims.shape[-1]
    f = filters.shape[0]
    nf = filters.shape[-1]
    row = 32 // f
    clown = 32 // f
    n_p = row * clown

    filters_torch = torch.tensor(filters, dtype=torch.float32, requires_grad=True)
    W1 = torch.tensor(network_params['W'][0], dtype=torch.float32, requires_grad=True)
    b1 = torch.tensor(network_params['b'][0], dtype=torch.float32, requires_grad=True)
    W2 = torch.tensor(network_params['W'][1], dtype=torch.float32, requires_grad=True)
    b2 = torch.tensor(network_params['b'][1], dtype=torch.float32, requires_grad=True)

    apply_relu = torch.nn.ReLU()
    apply_softmax = torch.nn.Softmax(dim=0)

    conv_outputs = []

    for i in range(n):
        single_img = X_torch[:, :, :, i]
        patches = []
        for j in range(row):
            for k in range(clown):
                patch = single_img[j*f:(j+1)*f, k*f:(k+1)*f, :]
                patches.append(patch.reshape(-1))

        patches = torch.stack(patches, dim=1)
        patches = patches.permute(1, 0)

        filters_flat = filters_torch.reshape(-1, nf)
        conv = patches @ filters_flat
        conv = apply_relu(conv)
        conv_outputs.append(conv.reshape(-1))
    conv_outputs = torch.stack(conv_outputs, dim=1)

    S1 = W1 @ conv_outputs + b1
    X1 = apply_relu(S1)
    S2 = W2 @ X1 + b2
    P = apply_softmax(S2)
    print("P_torch shape:", P.shape)
    print("P_torch:", P[:, 0])

    Y = torch.from_numpy(y).long()
    Y_labels = torch.argmax(Y, dim=0)
    loss = torch.mean(-torch.log(P[Y_labels , torch.arange(n)]))
    print("Loss in pytorch:", loss)

    loss.backward()

    grads = {}
    grads['W'] = [W1.grad.numpy(), W2.grad.numpy()]
    grads['b'] = [b1.grad.numpy(), b2.grad.numpy()]
    grads['Fs_flat'] = filters_torch.grad.reshape(-1, nf).numpy()

    return grads

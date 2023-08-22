import torch as tc
def remove_small(tensor_):
    d1 = tensor_.shape[0]
    d2 = tensor_.shape[-1]
    new_tensor = tc.clone(tensor_)
    for i in range(d1):
        for j in range(d2):
            tensor_real, tensor_imag = tc.real(tensor_[i][j]), tc.imag(tensor_[i][j])
            if tensor_real < 1e-16: new_tensor[i][j] = tensor_[i][j] + tc.tensor(1e-6)
            if tensor_imag < 1e-16: new_tensor[i][j] = tensor_[i][j] + tc.tensor(1e-6j)
    print('1')
    return new_tensor

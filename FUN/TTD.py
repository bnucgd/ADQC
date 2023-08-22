import torch as tc

'''将高阶张量进行严格TT分解'''
# nq = 16
# d = 2  # 物理指标的维数
# tensor = tc.randn([2] * nq)  # 建立一个nq比特的量子系统

dtype=tc.complex128
def TT_Decomposition(input_tensor):
    nq = len(input_tensor.shape)
    d = input_tensor.shape[0]
    psi = [None for _ in range(nq)]
    M_1 = input_tensor.reshape(input_tensor.shape[0], -1)
    u, s, v_dagger = tc.linalg.svd(M_1, full_matrices=False)
    s = tc.diag(s).to(dtype=dtype)
    psi[0] = u.reshape(1, d, -1)
    M_2 = s.mm(v_dagger).reshape([psi[0].shape[-1] * d] + [d] * (nq - 2))
    M_2_ =M_2.reshape(psi[0].shape[-1] * d, -1)

    for i in range(1, nq - 1):
        u, s, v_dagger = tc.linalg.svd(M_2_, full_matrices=False)
        s = tc.diag(s).to(dtype=dtype)
        psi[i] = u.reshape(psi[i-1].shape[-1], d, -1)
        M_2 = s.mm(v_dagger).reshape([psi[i].shape[-1] * d] + [d] * (nq - 2 - i))
        M_2_ = M_2.reshape(psi[i].shape[-1] * d, -1)

    psi[nq - 1] = M_2.reshape(psi[nq - 2].shape[-1], d, 1)
    return psi


# psi = TT_Decomposition(tensor)
# tensor_ = full_tensor(psi)
# tensor_ = tc.squeeze(tensor_)
# print('MPS与全局张量之差的二范数：', (tensor_ - tensor).norm().item())
#
# print('虚拟指标的维数:')
# virtual_index = list()
# for i in range(len(psi) - 1):
#     virtual_index.append(psi[i].shape[-1])
# print(virtual_index)

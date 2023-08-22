import math
import torch as tc
import matplotlib.pyplot as plt
from FUN import TTD
from torch import optim
from FUN.orthogonalization import orthogonalization, full_tensor
from Library.MatrixProductState import inner_product
import numpy as np
from Library.QuantumState import state_ghz, state_all_up
from FUN.TEBD2 import U_tensor, time_envolve

def latent_to_unitary(latent_paras):
    '''由一个1 * 16 的参数张量生成一个四阶张量（二体幺正门）'''
    latent_matrix = latent_paras.reshape(4, 4)
    P, _, Q_dagger = tc.linalg.svd(latent_matrix, full_matrices=True)
    unitary_gate = P.mm(Q_dagger).reshape(2, 2, 2, 2)
    return unitary_gate


def act_on_state(state, gate, left_to_right=True, vd = None):
    assert len(state) - 1 == len(gate)
    if left_to_right:
        for i in range(len(state) - 1):
            B = tc.einsum('ijst, asb, btd -> aijd', gate[i], state[i], state[i + 1])  # 缩并得到四阶张量
            B_ = B.reshape(state[i].shape[0] * state[i].shape[1], -1)  # 矩阵化
            u, lm, v_dagger = tc.linalg.svd(B_, full_matrices=True)  # 进行SVD分解
            lm = tc.tensor([i for i in lm if i >= 1e-16])
            lm_ = tc.diag(lm.to(dtype=dtype))
            u, v_dagger = u[:, :len(lm)].to(dtype=dtype), v_dagger[:len(lm), :].to(dtype=dtype)
            if vd != None:
                d = min(vd, len(lm))
                lm = lm[:d]
                lm_ = tc.diag(lm).to(dtype=dtype)
                state[i] = u[:, :d].reshape(state[i].shape[0], state[i].shape[1], d)  # u替代原来的左等距矩阵
                state[i + 1] = lm_.mm(v_dagger[:d, :]).reshape(d, state[i + 1].shape[1], state[i + 1].shape[2])
                state[i + 1] = state[i + 1] / state[i + 1].norm()
            else:
                state[i] = u.reshape(state[i].shape[0], state[i].shape[1], -1)  # u替代原来的左等距矩阵
                state[i + 1] = lm_.mm(v_dagger).reshape(-1, state[i + 1].shape[1], state[i + 1].shape[2])
                state[i + 1] = state[i + 1] / state[i + 1].norm()
    else:
        for i in range(len(state) - 1, 0, -1):
            B = tc.einsum('ijst, asb, btd -> aijd', gate[i - 1], state[i - 1], state[i])
            B_ = B.reshape(state[i - 1].shape[0] * state[i - 1].shape[1], -1)
            u, lm, v_dagger = tc.linalg.svd(B_, full_matrices=True)
            lm = tc.tensor([i for i in lm if i >= 1e-16])
            lm_ = tc.diag(lm.to(dtype=dtype))
            u, v_dagger = u[:, :len(lm)].to(dtype=dtype), v_dagger[:len(lm), :].to(dtype=dtype)
            if vd != None:
                d = min(vd, len(lm))
                lm = lm[:d]
                lm_ = tc.diag(lm).to(dtype=dtype)
                state[i] = v_dagger[:d, :].reshape(d, state[i].shape[1], state[i].shape[2])
                state[i - 1] = u[:, :d].mm(lm_).reshape(state[i - 1].shape[0], state[i - 1].shape[1], d)
                state[i - 1] = state[i - 1] / state[i - 1].norm()
            else:
                state[i] = v_dagger.reshape(-1, state[i].shape[1], state[i].shape[2])
                state[i - 1] = (u @ lm_).reshape(state[i - 1].shape[0], state[i - 1].shape[1], -1)
                state[i - 1] = state[i - 1] / state[i - 1].norm()
    return state
    # return state_update


def latent_initial_paras(gate_num):
    unit_matrix = np.eye(4) + 1e-05 * np.random.randn(4, 4)
    unit_matrix = tc.from_numpy(unit_matrix)
    stack_array = np.stack([unit_matrix for _ in range(gate_num)])
    return stack_array



'''生成目标态'''
dtype = tc.complex128
nq = 6
d = 2
# seed = 42
# tc.manual_seed(seed)
# psi_target = tc.randn([d] * nq, dtype=dtype)
# tc.seed()
# psi_target /= psi_target.norm()
# psi = TTD.TT_Decomposition(psi_target)



psi = [tc.randn(1, d, 3, dtype=tc.complex128)] + [tc.randn(3, d, 3, dtype=tc.complex128) for _ in range(nq - 2)] +\
      [tc.randn(3, d, 1, dtype=tc.complex128)]
t, tau_ = 1000, 1
err = 1
layers = 0
while err > 1e-6:
    a = full_tensor(psi)
    U_tau, tau = U_tensor(tau_, Jx=1, Jy=1, Jz=0, if_i=False)  # 建立虚时演化算符和虚时间演化算符切片（Jx， Jy， Jz）
    layers = time_envolve(t, tau, U_tau, psi)  # 进行虚时演化，得到基态
    layers += layers
    # time_envolve(t, tau, U_tau, phi)  # 进行虚时演化，得到基态, 不裁剪
    b = full_tensor(psi)
    t /= 10
    tau_ /= 10
    err = ((a - b).norm().item()) / a.norm().item()
    # print('相对误差:', err)
ground_state1, psi_target = tc.squeeze(b), psi
print(psi)
print('################    收敛，得到基态波函数     ####################')

'''生成初态'''
# phi = [tc.randn(1, d, 3, dtype=dtype)] + [tc.randn(3, d, 3, dtype=dtype) for _ in range(nq - 2)] \
#       + [tc.randn(3, d, 1, dtype=dtype)]
#
# phi_initial = state_ghz(nq).to(dtype=dtype)
phi_initial = state_all_up(nq, d).to(dtype=dtype)
phi = TTD.TT_Decomposition(phi_initial)
orthogonalization(phi, tar_center_pos=0, if_normalization=True)  # 对初态MPS进行归一化处理（正交形式）
'''建立优化器'''
iter_times, lr = 4000, 1e-3
layers = 3  # 量子门的作用层数
# gate_num = (nq - 1) * layers
gate_num = nq - 1


# tc.autograd.set_detect_anomaly(True)


for n_layers in range(layers):
    if n_layers == 0:
        '''不同的初始化方案'''
        var_paras = tc.randn(gate_num, 16, dtype=dtype, requires_grad=True)

    else:
        var_paras = tc.tensor(latent_initial_paras(gate_num), dtype=dtype, requires_grad=True)


    optimizer = optim.Adam([var_paras], lr=lr)
    '''开始迭代'''
    loss_list=[]
    for i in range(iter_times):
        '''建立初态MPS'''
        original_state = [tensor_.clone() for tensor_ in phi]
        '''生成量子门'''
        gates = [None for _ in range(gate_num)]
        for n in range(gate_num):
            gates[n] = latent_to_unitary(var_paras[n, :])
        '''将量子门作用到量子态上'''
        original_state = act_on_state(state=original_state, gate=gates, left_to_right=((n_layers % 2) == 0))
        '''构建loss'''
        loss = - inner_product(psi, original_state) / nq
        # loss = - inner_product(psi, original_state)
        # loss = 1 - full_tensor(psi).conj().flatten().dot(full_tensor(original_state).flatten()).norm()
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        '''打印loss'''
        if (i+1) % 20 == 0:
            print('第%d层量子门, 第%d次迭代, loss=' % ((n_layers + 1), (i + 1)), loss.item(), '保真度：', math.exp(-loss * nq))

        loss_list.append(loss.item())
    plt.plot(range(iter_times), loss_list)
    plt.show()
    phi = original_state





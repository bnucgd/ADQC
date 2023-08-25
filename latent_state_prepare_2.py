import math
import torch as tc
from FUN import TTD
from torch import optim
from FUN.orthogonalization import orthogonalization, full_tensor, inner_product
# from Library.MatrixProductState import inner_product
import numpy as np
from Library.QuantumState import state_ghz
from FUN.TEBD2 import U_tensor, time_envolve
import matplotlib.pyplot as plt


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
            u, lm, v_dagger = tc.linalg.svd(B_, full_matrices=False)  # 进行SVD分解
            lm = tc.tensor([i for i in lm if i >= 1e-15])
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
            u, lm, v_dagger = tc.linalg.svd(B_, full_matrices=False)
            lm = tc.tensor([i for i in lm if i >= 1e-15])
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
                state[i - 1] = u.mm(lm_).reshape(state[i - 1].shape[0], state[i - 1].shape[1], -1)
                state[i - 1] = state[i - 1] / state[i - 1].norm()
    return state
    # return state_update


def latent_initial_paras(gate_num):
    stack_array = np.stack([(np.eye(4) + 1e-05 * np.random.randn(4, 4)) for _ in range(gate_num)])
    return stack_array


def enable_gradient_tracking(lst):
    for item in lst:
        if isinstance(item, tc.Tensor):
            item.requires_grad_()


# tc.autograd.set_detect_anomaly(True)
'''生成目标态'''
dtype = tc.complex128
eps = 1e-6
vd = 8
nq = 16
d = 2
'''生成目标态'''
psi = [tc.randn(1, d, 3, dtype=dtype)] + [tc.randn(3, d, 3, dtype=dtype) for _ in range(nq - 2)] +\
      [tc.randn(3, d, 1, dtype=dtype)]
t, tau_ = 1000, 1
err = 1
while err > 1e-6:
    a = full_tensor(psi)
    U_tau, tau = U_tensor(tau_, Jx=1, Jy=1, Jz=0, if_i=False)  # 建立虚时演化算符和虚时间演化算符切片（Jx， Jy， Jz）
    time_envolve(t, tau, U_tau, psi)  # 进行虚时演化，得到基态
    b = full_tensor(psi)
    t /= 10
    tau_ /= 10
    err = ((a - b).norm().item()) / a.norm().item()

print('################    收敛，得到基态波函数     ####################')
'''生成初态'''
phi_initial = state_ghz(nq).to(dtype=dtype)
# phi_initial = state_all_up(nq, d).to(dtype=dtype)
phi = TTD.TT_Decomposition(phi_initial)
orthogonalization(phi, tar_center_pos=0, if_normalization=True)  # 对初态MPS进行归一化处理（正交形式）
phi_ = [tensor_ for tensor_ in phi]
print('生成初态')

'''建立优化器'''
iter_times, lr = 4000, 1e-3
layers = 3  # 量子门的作用层数
gate_num = nq - 1


gates_history = [None for _ in range(layers)]
for n_layers in range(layers):
    if n_layers == 0:
        '''不同的初始化方案'''
        var_paras = tc.randn(gate_num, 16, dtype=dtype, requires_grad=True)
    else:
        var_paras = tc.tensor(latent_initial_paras(gate_num), dtype=dtype, requires_grad=True)

    optimizer = optim.Adam([var_paras], lr=lr)
    '''开始迭代'''
    loss_list = []
    for i in range(iter_times):
        '''建立初态MPS'''
        original_state = [tensor_.clone() for tensor_ in phi]
        '''生成量子门'''
        gates = [None for _ in range(gate_num)]
        for n in range(gate_num):
            gates[n] = latent_to_unitary(var_paras[n, :])
        '''将量子门作用到量子态上'''
        # original_state = act_on_state(state=original_state, gate=gates, left_to_right=((n_layers % 2) == 0))
        original_state = act_on_state(state=original_state, gate=gates, left_to_right=((n_layers % 2) == 0))
        '''构建loss'''
        loss = - tc.log(inner_product(psi, original_state)) / nq
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        '''打印loss'''
        if (i + 1) % 20 == 0:
            print('第%d层量子门, 第%d次迭代, loss=' % ((n_layers + 1), (i + 1)), loss.item(), '保真度：',
                  math.exp(-loss * nq))

        loss_list.append(loss.item())

    plt.plot(loss_list, 'r-o')
    # plt.title('第%d层量子门' % (n_layers + 1))
    plt.show()
    # disable_gradient_tracking(var_paras)
    phi = original_state
    '''将生成的门保存下来'''
    gates_history[n_layers] = [g.detach() for g in gates]
    for n in range(n_layers + 1):
        if n_layers == 0:continue
        enable_gradient_tracking(gates_history[n])
        optimizer_1 = tc.optim.Adam(gates_history[n], lr=lr)
        loss_list = []
        '''将量子门作用到量子态上'''
        for num in range(iter_times):
            original_state_ = [tensor_.clone() for tensor_ in phi_]
            for m in range(n+1):
                act_on_state(state=original_state_, gate=gates_history[m],
                                              left_to_right=((n % 2) == 0))
            '''构建loss'''
            loss_1 = - tc.log(inner_product(psi, original_state_)) / nq
            loss_1.backward(retain_graph=True)
            optimizer_1.step()
            optimizer_1.zero_grad()
            '''打印loss'''
            if (num + 1) % 20 == 0:
                print('共%d层, 重新优化第%d层量子门, 第%d次迭代, loss=' % ((n_layers + 1), (n + 1), (num + 1)),
                      loss_1.item(), '保真度：', math.exp(-loss_1 * nq))
            loss_list.append(loss_1.item())
        plt.plot(loss_list, 'r-o')
        # plt.title('共%d层，重新优化第%d层量子门' % ((n_layers + 1), (n + 1)))
        plt.show()

        phi = original_state_
    # phi = original_state_

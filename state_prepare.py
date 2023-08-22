import copy
from torch import optim
import torch as tc
from Library.QuantumState import state_all_up
import numpy as np

'''生成目标态'''
nq = 4  # 比特数
d = 2  # 物理指标的维数
tar_state = tc.randn([d] * nq, dtype=tc.complex128)
tar_state /= tar_state.norm()






def latent_unitary(latent_gate=None):
    if latent_gate == None:
        latent_gate = tc.randn([d] * 4)
    latent_gate = latent_gate.reshape(d ** 2, -1)
    P, _, Q_dagger = tc.linalg.svd(latent_gate)
    unitary_matrix = P.mm(Q_dagger)
    unitary_gate = unitary_matrix.reshape([d] * 4)
    return unitary_gate


def act_opertor(unitary_gate, state, pos):
    small = [chr(i) for i in range(97, 123)]  # 生成一个字母表
    big = [chr(i) for i in range(65, 91)]
    state_index = ''
    for i in small[:nq]:
        state_index += i
    opertor_index = ''
    for i in big[:nq]:
        opertor_index += i

    unitary_gate_index = state_index[pos[0]] + state_index[pos[1]] + opertor_index[pos[0]] + opertor_index[pos[1]]
    state_index_ =copy.deepcopy(list(state_index))
    state_index_[pos[0]], state_index_[pos[1]] = state_index_[pos[0]].upper(), state_index_[pos[1]].upper()
    fin_state_index = ''
    for i in state_index_:
        fin_state_index += i
    state = tc.einsum('{},{} -> {} '.format(unitary_gate_index, state_index, fin_state_index),
                        unitary_gate, state)
    return state


iter_time, lr = 5000, 0.01
array_ = np.eye(4) + 1e-5 * np.random.random((4, 4))
ini_latent = tc.tensor(array_, dtype=tc.complex128, requires_grad=True)
var_paras = [ini_latent for _ in range(nq - 1)]
optimizer = optim.Adam(var_paras, lr=lr)

for i in range(iter_time):
    '''建立初态'''
    state_original = state_all_up(n_qubit=nq, d=d)
    state_original = state_original.to(dtype=tc.complex128)
    unitary_gates = [None for _ in range(nq - 1)]
    for j in range(nq - 1):  # 生成幺正门
        unitary_gates[j] = latent_unitary(var_paras[j])

    for k in range(nq - 1):  # 开始作用
        state_original = act_opertor(unitary_gates[k], state=state_original, pos=[k, k + 1])
    loss = 1 - (state_original.flatten().dot(tar_state.flatten().conj()).norm())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (i + 1) % 20 == 0:
        print('第%d次迭代, loss：' % (i+1), loss)






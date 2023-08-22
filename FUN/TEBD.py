import torch as tc
from Library.MathFun import pauli_operators
from Library.MatrixProductState import MPS_basic


op = pauli_operators()


def U_ij_matrix(tau, if_i, Jx=1, Jy=1, Jz=1):  # tau即时间切片大小, 得到直积表象下的矩阵形式
    H_ij = Jx * tc.kron(op['x'], op['x']) + \
           Jy * tc.kron(op['y'], op['y']) + \
           Jz * tc.kron(op['z'], op['z'])
    if if_i:  # 时间演化算符
        U_ij = tc.matrix_exp(complex(0, -1) * tau * H_ij)
    else:  # 有限温度密度矩阵算符（对应虚时演化）
        U_ij = tc.matrix_exp(complex(-1, 0) * tau * H_ij)
    return U_ij


def U_tensor (tau, Jx, Jy, Jz,if_i):  # 得到时间演化算符切片的张量形式
    # base11 = tc.tensor([1, 0, 0, 0])
    # base10 = tc.tensor([0, 1, 0, 0])
    # base01 = tc.tensor([0, 0, 1, 0])
    # base00 = tc.tensor([0, 0, 0, 1])
    base = [tc.tensor([1, 0], dtype=tc.complex128), tc.tensor([0, 1], dtype=tc.complex128)]
    U_ij = U_ij_matrix(tau, if_i, Jx, Jy, Jz)
    U = tc.zeros(2, 2, 2, 2, dtype=tc.complex128)
    for i in range(2):
        for j in range(2):
            for a in range(2):
                for b in range(2):
                    bra = tc.kron(base[i], base[j])
                    ket = tc.kron(base[a], base[b])
                    U[i][j][a][b] = tc.einsum('a, ab, b ->', bra, U_ij, ket)
    return U, tau


def left_right(U_tau, psi):  # 时间演化算符切片从左向右按阶梯式作用(输入算符切片和要作用的类)
    for i in range(len(psi.tensors) - 1):
        psi.center_orthogonalization(c=i+1, normalize=True)  # 第一步，首先正交化
        B = tc.einsum('ijst, asb, btd -> aijd', U_tau, psi.tensors[i], psi.tensors[i+1])  # 缩并得到四阶张量
        B_ = B.reshape(psi.tensors[i].shape[0] * psi.para['d'], -1)  # 矩阵化
        u, lm, v_dagger = tc.linalg.svd(B_, full_matrices=False)  # 进行SVD分解
        lm_ = tc.diag(lm).to(dtype=tc.complex128)
        u, v_dagger = u.to(dtype=tc.complex128), v_dagger.to(dtype=tc.complex128)
        psi.tensors[i] = u.reshape(psi.tensors[i].shape[0], psi.tensors[i].shape[1], -1)  # u替代原来的左等距矩阵
        psi.tensors[i+1] = lm_.mm(v_dagger).reshape(-1, psi.tensors[i+1].shape[1], psi.tensors[i+1].shape[2])


def right_left(U_tau, psi):  # 时间演化算符切片从右向左按阶梯式作用
    for i in range(len(psi.tensors) - 1, 0, -1):
        psi.center_orthogonalization(c=i-1, normalize=True)
        B = tc.einsum('ijst, asb, btd -> aijd', U_tau, psi.tensors[i-1], psi.tensors[i])
        B_ = B.reshape(psi.tensors[i-1].shape[0] * psi.para['d'], -1)
        u, lm, v_dagger = tc.linalg.svd(B_, full_matrices=False)
        lm_ = tc.diag(lm).to(dtype=tc.complex128)
        u, v_dagger = u.to(dtype=tc.complex128), v_dagger.to(dtype=tc.complex128)
        psi.tensors[i] = v_dagger.reshape(-1, psi.tensors[i].shape[1], psi.tensors[i].shape[2])
        psi.tensors[i-1] = u.mm(lm_).reshape(psi.tensors[i-1].shape[0], psi.tensors[i-1].shape[1], -1)


def time_envolve(K,tau,U_tau, psi):  # K为总时间， tau为时间切片， U_tau为四阶张量， psi为MPS_Basic类
    for times in range(int(K // tau)):
        if times % 2 == 0:
            left_right(U_tau, psi)
        else:
            right_left(U_tau, psi)


# # ########    进行虚时演化得到基态    ############## #
# para0 = {
#             'length': 10,
#             'd': 2,
#             'chi': 3,
#             'boundary': 'open',
#             'device': None,
#             'dtype': tc.complex128
#         }
# phi = MPS_basic(para=para0)  # 建立一个随机MPS
# t = 1000
# tau_ = 1
# err = 1
# while err > 1e-6:
#     a = phi.full_tensor()
#     U_tau, tau = U_tensor(tau_, Jx=1, Jy=1, Jz=0, if_i=False)  # 建立虚时演化算符和虚时间演化算符切片
#     time_envolve(t, tau, U_tau, phi)  # 进行虚时演化，得到基态
#     b = phi.full_tensor()
#     t /= 10
#     tau_ /= 10
#     err = ((a - b).norm().item()) / a.norm().item()
#     # print(err)
# print('################    收敛，得到基态     ####################')
# groud_state, groud_state_ = b, phi.tensors
# print(groud_state)
#
#
#
# # ########    进行时间演化    ############## #
# tau_, t = 0.1, 0.3
# U_tau, tau = U_tensor(tau_, Jx=1, Jy=1, Jz=0, if_i=True)  # 建立时间演化算符和虚时间演化算符切片
# time_envolve(t, tau, U_tau, phi)
# # phi.normalize()  # 归一化
# # excited_state, excited_state_ = phi.full_tensor(), phi.tensors
# print('###############    演化的效果   ###############')
# print(phi.full_tensor())
# phi.check_center_orthogonality()
# print(phi.norm_square())
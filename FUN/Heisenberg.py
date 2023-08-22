import torch as tc
from Library.MathFun import pauli_operators
from scipy.sparse.linalg import eigsh
from orthogonalization import orthogonalization, full_tensor
from TEBD2 import U_tensor, time_envolve
from Library.ED import ED_ground_state

'''生成直积表象下的哈密顿量'''
op = pauli_operators()
id = tc.eye(2, dtype=tc.complex128)


class Heisenberg_Hamiltonians():

    def __init__(self, nq, para=None):
        self.para = dict()
        self.input_paras(para)
        self.nq = nq

    def input_paras(self, para=None):
        para0 = {
            'Jx': 1,
            'Jy': 1,
            'Jz': 0,
            'dtype': tc.float64
        }
        if para is None:
            self.para = para0
        else:
            self.para = dict(para0, **para)

    def X(self):  # x方向的哈密顿量
        assert type(self.nq) == int
        Hx = 0
        if self.para['Jx'] != 0:
            for i in range(self.nq - 1):
                Hx += self.para['Jx'] * tc.einsum('ab, bc->ac', self.Sx(i), self.Sx(i + 1))
        return Hx

    def Y(self):
        assert type(self.nq) == int
        Hy = 0
        if self.para['Jy'] != 0:
            for i in range(self.nq - 1):
                Hy += self.para['Jy'] * tc.einsum('ab, bc->ac', self.Sy(i), self.Sy(i + 1))
        return Hy

    def Z(self):
        assert type(self.nq) == int
        Hz = 0
        if self.para['Jz'] != 0:
            for i in range(self.nq - 1):
                Hz += self.para['Jz'] * tc.einsum('ab, bc->ac', self.Sz(i), self.Sz(i + 1))
        return Hz

    def H(self):  # 哈密顿量
        return self.X() + self.Y() + self.Z()

    def U(self, t):  # 时间演化算符
        return tc.matrix_exp(complex(0, -1) * t * self.H())

    def Sx(self, position):  # 福克空间的Sx算符
        length = self.nq
        assert length - 1 >= position
        Sx_ = tc.tensor(1, dtype=tc.complex128)
        for i in range(length):
            if i == position:
                Sx_ = tc.kron(Sx_, op['x'])
            else:
                Sx_ = tc.kron(Sx_, id)
        return Sx_

    def Sy(self, position):
        length = self.nq
        assert length - 1 >= position
        Sy_ = tc.tensor(1, dtype=tc.complex128)
        for i in range(length):
            if i == position:
                Sy_ = tc.kron(Sy_, op['y'])
            else:
                Sy_ = tc.kron(Sy_, id)
        return Sy_

    def Sz(self, position):
        length = self.nq
        assert length - 1 >= position
        Sz_ = tc.tensor(1, dtype=tc.complex128)
        for i in range(length):
            if i == position:
                Sz_ = tc.kron(Sz_, op['z'])
            else:
                Sz_ = tc.kron(Sz_, id)
        return Sz_






# ########  严格对角化(直积表象)   #############
# a = Heisenberg_Hamiltonians(nq=5, para=None)
# # print(a.H())
# ground_energy1, ground_state1 = eigsh(a.H().numpy(), k=1, which='SA')
# print('本征值分解得到的基态能量：', ground_energy1)
# # print('基态波函数：', ground_state1)



#
# ######## 线性算子法（严格对角化）##########
# hamilt = Heisenberg_Hamiltonians(nq=2, para=None)
# ground_energy2, ground_state2 = ED_ground_state(hamilt.H().reshape(2, 2, 2, 2),
#                            pos=[[i, i+1] for i in range(15)])
#
# print('线性算子法：基态能量：', ground_energy2[0])
#
#
#
#
#
# ############## TEBD  ##################
#
# print('*****虚时演化*******')
# # para1 = {
# #             'length': 5,
# #             'd': 2,
# #             'chi': 3,
# #             'boundary': 'open',
# #             'device': None,
# #             'dtype': tc.complex128
# #         }
# # phi = MPS_basic(para=para1)  # 建立一个随机MPS
#
# length = 16  # MPS的长度
# phi = [tc.randn(1, 2, 3, dtype=tc.complex128)] + [tc.randn(3, 2, 3, dtype=tc.complex128) for _ in range(length - 2)]\
#       + [tc.randn(3, 2, 1, dtype=tc.complex128)]
# orthogonalization(phi, tar_center_pos=0)
# t = 1000
# tau_ = 1
# err = 1
# while err > 1e-6:
#     a = full_tensor(phi)
#     U_tau, tau = U_tensor(tau_, Jx=1, Jy=1, Jz=0, if_i=False)  # 建立虚时演化算符和虚时间演化算符切片
#     time_envolve(t, tau, U_tau, phi, vd=2)  # 进行虚时演化，得到基态
#     # time_envolve(t, tau, U_tau, phi)  # 进行虚时演化，得到基态, 不裁剪
#     b = full_tensor(phi)
#     t /= 10
#     tau_ /= 10
#     err = ((a - b).norm().item()) / a.norm().item()
#     print('相对误差:', err)
# print('################    收敛，得到基态     ####################')
# ground_state3, ground_state3_ = tc.squeeze(b), phi
#
# H_ij = tc.kron(op['x'], op['x']) + \
#        tc.kron(op['y'], op['y'])
# H_ij = H_ij.reshape(2, 2, 2, 2)
#
#
#
#
# ground_energy3 = 0
# bra = ground_state3.conj()
# ket = ground_state3
# small = [chr(i) for i in range(97, 123)]  # 生成一个字母表
# big = [chr(i) for i in range(65, 91)]
#
# ket_index = ''
# for i in small[:length]:
#     ket_index += i
# bra_index = ''
# for i in big[:length]:
#     bra_index += i
#
#
# for i in range(length - 1):
#     H_ij_index = small[i] + small[i + 1] + big[i] + big[i + 1]
#     ket_index_ = list(ket_index)
#     for j in range(len(ket_index)):
#         if ket_index_[j] not in H_ij_index:
#             ket_index_[j] = ket_index_[j].upper()
#     ket_index_ = ''.join(ket_index_)
#     ground_energy3 += tc.einsum('{},{},{} -> '.format(bra_index, H_ij_index, ket_index_), bra, H_ij, ket)
#
#
# print('虚时演化得到的基态能量：', ground_energy3)
# assert tc.imag(ground_energy3) < 1e-10
# ground_energy3 = tc.real(ground_energy3).item()
# print('虚时演化得到的基态能量：', ground_energy3)
#

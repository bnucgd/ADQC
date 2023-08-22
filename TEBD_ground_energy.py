import torch as tc
from FUN.TTD import TT_Decomposition
from FUN.TEBD2 import U_tensor, time_envolve
from FUN.orthogonalization import full_tensor
from Library.MathFun import pauli_operators
from Library.ED import ED_ground_state
import matplotlib.pyplot as plt
import matplotlib


'''随机建立一个nq比特的量子系统'''
nq = 16
d = 2  # 物理指标的维数
# input_tensor = tc.randn([d] * nq, dtype=tc.complex128)
#
#
# '''对全局张量进行严格TT分解,得到中心在-1处的正交形式的MPS'''
# psi = TT_Decomposition(input_tensor)
#
# vd_list = list()  # 打印MPS的虚拟维数
# for i in range(len(psi) - 1):
#     vd_list.append(psi[i].shape[-1])
# # print('MPS的虚拟指标维数：', vd_list)

psi = [tc.randn(1, d, 3, dtype=tc.complex128)] + [tc.randn(3, d, 3, dtype=tc.complex128) for _ in range(nq - 2)] +\
      [tc.randn(3, d, 1, dtype=tc.complex128)]



'''进行虚时演化'''
ground_energy, times = [], []
chi = [2, 4, 8, 16, 32]  # 设定的虚拟维数上限
for vd_ in chi:
    # vd_ = 2  # 裁剪后的虚拟维数上限
    t, tau_ = 1000, 1
    err = 1
    layers = 0
    while err > 1e-6:
        a = full_tensor(psi)
        U_tau, tau = U_tensor(tau_, Jx=1, Jy=1, Jz=0, if_i=False)  # 建立虚时演化算符和虚时间演化算符切片（Jx， Jy， Jz）
        layers = time_envolve(t, tau, U_tau, psi, vd=vd_)  # 进行虚时演化，得到基态
        layers += layers
        # time_envolve(t, tau, U_tau, phi)  # 进行虚时演化，得到基态, 不裁剪
        b = full_tensor(psi)
        t /= 10
        tau_ /= 10
        err = ((a - b).norm().item()) / a.norm().item()
        # print('相对误差:', err)

    times.append(layers)
    ground_state1, ground_state1_ = tc.squeeze(b), psi
    # print('################    收敛，得到基态波函数     ####################')

    '''利用时间演化得到的基态波函数，计算基态能级'''
    op = pauli_operators()
    H_ij = tc.kron(op['x'], op['x']) + tc.kron(op['y'], op['y'])
    H_ij = H_ij.reshape(2, 2, 2, 2)  # 定义局域哈密顿量
    bra = ground_state1.conj()
    ket = ground_state1

    # 为einsum函数的指标作准备
    small = [chr(i) for i in range(97, 123)]  # 生成一个字母表
    big = [chr(i) for i in range(65, 91)]
    ket_index = ''
    for i in small[:nq]:
        ket_index += i
    bra_index = ''
    for i in big[:nq]:
        bra_index += i

    #  将算符作用到ket上，并与bra缩并
    ground_energy1 = 0
    for i in range(nq - 1):
        H_ij_index = small[i] + small[i + 1] + big[i] + big[i + 1]
        ket_index_ = list(ket_index)
        for j in range(len(ket_index)):  # 将bra和ket中不与局域哈密顿量算符缩并的指标变成相同的字母，以便后续缩并掉
            if ket_index_[j] not in H_ij_index:
                ket_index_[j] = ket_index_[j].upper()
        ket_index_ = ''.join(ket_index_)
        ground_energy1 += tc.einsum('{},{},{} -> '.format(bra_index, H_ij_index, ket_index_), bra, H_ij, ket)

    # print('虚时演化得到的基态能量：', ground_energy1)  # 虚部不严格为零
    assert tc.imag(ground_energy1) < 1e-15  # 虚部很小时，认为其为零
    ground_energy1 = tc.real(ground_energy1).item()  # 取在虚部可忽略的情况下取实
    # print('虚时演化得到的基态能量：', ground_energy1)
    ground_energy.append(ground_energy1)


'''线性算子法（严格对角化）'''
# 调用了老师的库函数

ground_energy2, ground_state2 = ED_ground_state(H_ij.reshape(2, 2, 2, 2),
                           pos=[[i, i+1] for i in range(nq - 1)])

print('线性算子法：基态能量：', ground_energy2[0])


# print('MPS的虚拟指标维数：', vd_list)
print('设定的裁剪后的最高维数chi：', chi)
print('演化过程中作用的层数：', times)

print('裁剪后的基态能量：', ground_energy)
accuracy = ground_energy / ground_energy2[0]
accuracy = ['%.10f' % i for i in accuracy]
print('裁剪后的精度：', accuracy)
plt.plot(chi, accuracy, 'r-o')
plt.xlabel('虚拟维数')
plt.ylabel('精度')
plt.xticks(chi)
matplotlib.rcParams['font.sans-serif'] = ['SimHei']     # 显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
plt.show()

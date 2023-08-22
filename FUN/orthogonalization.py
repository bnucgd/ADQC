import torch as tc
from Library.MatrixProductState import check_center_orthogonality

'''正交化MPS或移动正交中心'''
dtype=tc.complex128


def orthogonalization(psi, tar_center_pos, center_pos=None, if_normalization=True):  # psi为一TT形式, tar_center_pos是中心张量的位置
    if center_pos == None:  # 输入的MPS不是中心正交化的
        for i in range(tar_center_pos):
            four_tensor = tc.einsum('abc, cde -> abde', psi[i], psi[i + 1])
            matrix_ = four_tensor.reshape(psi[i].shape[0] * psi[i].shape[1],
                                          psi[i + 1].shape[1] * psi[i + 1].shape[2])
            u, lm, v_dagger = tc.linalg.svd(matrix_, full_matrices=False)

            lm = tc.tensor([i for i in lm if i != 0])

            lm_= tc.diag(lm).to(dtype=dtype)
            u, v_dagger = u[:, :len(lm)].to(dtype=dtype), v_dagger[:len(lm), :].to(dtype=dtype)
            psi[i] = u.reshape(psi[i].shape[0], psi[i].shape[1], -1)
            psi[i + 1] = lm_.mm(v_dagger).reshape(-1, psi[i + 1].shape[1], psi[i + 1].shape[2])

        for i in range(len(psi) - 1, tar_center_pos, -1):
            four_tensor = tc.einsum('abc, cde -> abde', psi[i - 1], psi[i])
            matrix_ = four_tensor.reshape(psi[i - 1].shape[0] * psi[i - 1].shape[1],
                                          psi[i].shape[1] * psi[i].shape[2])
            u, lm, v_dagger = tc.linalg.svd(matrix_, full_matrices=False)

            lm = tc.tensor([i for i in lm if i != 0])

            lm_= tc.diag(lm).to(dtype=dtype)
            u, v_dagger = u[:, :len(lm)].to(dtype=dtype), v_dagger[:len(lm), :].to(dtype=dtype)
            psi[i - 1] = u.mm(lm_).reshape(psi[i - 1].shape[0], psi[i - 1].shape[1], -1)
            psi[i] = v_dagger.reshape(-1, psi[i].shape[1], psi[i].shape[2])
        if if_normalization:
            psi[tar_center_pos] /= psi[tar_center_pos].norm()
    else:  # 输入的MPS是中心正交化的（改变中心位置）
        err = check_center_orthogonality(psi, center=center_pos, prt=False)  # 检查输入是否真的是中心正交化的MPS
        err = [i for i in err if i != None]
        err_av = sum(err) / len(err)
        if err_av < 1e-03:  # 检验是否真的是中心正交化的
            if tar_center_pos > center_pos:  # 中心位置向右移动
                for i in range(center_pos, tar_center_pos):
                    four_tensor = tc.einsum('abc, cde -> abde', psi[i], psi[i + 1])
                    matrix_ = four_tensor.reshape(psi[i].shape[0] * psi[i].shape[1],
                                                  psi[i + 1].shape[1] * psi[i + 1].shape[2])
                    u, lm, v_dagger = tc.linalg.svd(matrix_, full_matrices=False)

                    lm = tc.tensor([i for i in lm if i != 0])

                    lm_ = tc.diag(lm).to(dtype=dtype)
                    u, v_dagger = u[:, :len(lm)].to(dtype=dtype), v_dagger[:len(lm), :].to(dtype=dtype)
                    psi[i] = u.reshape(psi[i].shape[0], psi[i].shape[1], -1)
                    psi[i + 1] = lm_.mm(v_dagger).reshape(-1, psi[i + 1].shape[1], psi[i + 1].shape[2])
                    if if_normalization:
                        psi[tar_center_pos] /= psi[tar_center_pos].norm()
            else:  # 中心位置向左移动
                for i in range(center_pos, tar_center_pos, -1):
                    four_tensor = tc.einsum('abc, cde -> abde', psi[i - 1], psi[i])
                    matrix_ = four_tensor.reshape(psi[i - 1].shape[0] * psi[i - 1].shape[1],
                                                  psi[i].shape[1] * psi[i].shape[2])
                    u, lm, v_dagger = tc.linalg.svd(matrix_, full_matrices=False)
                    lm_ = tc.diag(lm).to(dtype=dtype)
                    u, v_dagger = u.to(dtype=dtype), v_dagger.to(dtype=dtype)
                    psi[i - 1] = u.mm(lm_).reshape(psi[i - 1].shape[0], psi[i - 1].shape[1], -1)
                    psi[i] = v_dagger.reshape(-1, psi[i].shape[1], psi[i].shape[2])
                    if if_normalization:
                        psi[tar_center_pos] /= psi[tar_center_pos].norm()

        else:
            print('err大于1e-03，中心位置不在%d' % center_pos)
            print(err_av)

    return center_pos


def full_tensor(psi):  # 输入一个MPS
    full_tensor = psi[0]
    for tensor in psi[1:]:
        full_tensor = tc.tensordot(full_tensor, tensor, [[-1], [0]])
    return full_tensor

# length = 16  # MPS的长度
# phi = [tc.randn(1, 2, 3, dtype=tc.complex128)] + [tc.randn(3, 2, 3, dtype=tc.complex128) for _ in range(length - 2)] \
#       + [tc.randn(3, 2, 1, dtype=tc.complex128)]
# orthogonalization(phi, tar_center_pos=2)
# check_center_orthogonality(phi, center=2, prt=True)
# orthogonalization(phi, tar_center_pos=4, center_pos=2)  # 正交中心在2
# orthogonalization(phi, tar_center_pos=4, center_pos=3)  # 正交中心在3
# # check_center_orthogonality(phi, center=4, prt=True)  # 用老师的库函数检验上述函数是否实现正交化的功能

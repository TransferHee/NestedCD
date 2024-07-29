from torch import nn
from torch.autograd import Function
import torch
import importlib
import os

chamfer_found = importlib.find_loader("mchamfer_3D20") is not None
if not chamfer_found:
    ## Cool trick from https://github.com/chrdiller
    print("Jitting Chamfer Modified 3D")
    cur_path = os.path.dirname(os.path.abspath(__file__))
    build_path = cur_path.replace('chamferM3D5', 'tmp')
    os.makedirs(build_path, exist_ok=True)

    from torch.utils.cpp_extension import load

    chamfer_3DM20 = load(name="mchamfer_3D20",
                        sources=[
                            "/".join(os.path.abspath(__file__).split('/')[:-1] + ["chamfer_cuda.cpp"]),
                            "/".join(os.path.abspath(__file__).split('/')[:-1] + ["chamferM3D.cu"]),
                        ], build_directory=build_path)
    print("Loaded JIT 3D CUDA Modified chamfer distance")

else:
    import mchamfer_3D20

    print("Loaded compiled 3D CUDA Modified chamfer distance")


# Chamfer's distance module @thibaultgroueix
# GPU tensors only
class chamfer_3DFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, dim = xyz1.size()
        assert dim == 3, "Wrong last dimension for the chamfer distance 's input! Check with .size()"
        _, m, dim = xyz2.size()
        assert dim == 3, "Wrong last dimension for the chamfer distance 's input! Check with .size()"
        device = xyz1.device

        device = xyz1.device

        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

        dist1 = dist1.to(device)
        dist2 = dist2.to(device)
        idx1 = idx1.to(device)
        idx2 = idx2.to(device)
        torch.cuda.set_device(device)

        chamfer_3DM20.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()
        device = graddist1.device

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        gradxyz1 = gradxyz1.to(device)
        gradxyz2 = gradxyz2.to(device)
        chamfer_3DM20.backward(
            xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2
        )
        return gradxyz1, gradxyz2


class chamfer_3DDist(nn.Module):
    def __init__(self):
        super(chamfer_3DDist, self).__init__()

    def forward(self, input1, input2):
        input1 = input1.contiguous()
        input2 = input2.contiguous()
        return chamfer_3DFunction.apply(input1, input2)


if __name__ == '__main__':
    import numpy as np
    import torch

    torch.manual_seed(1)

    np.random.seed(1)

    x1 = torch.rand(1, 2048, 3).cuda()
    x2 = torch.rand(1, 2048, 3).cuda()

    ff = chamfer_3DDist()
    dist1, dist2, idx1, idx2 = ff.forward(x1, x2)

    idx1 = idx1.cpu().numpy()[0]
    idx2 = idx2.cpu().numpy()[0]

    cnt = 0
    for i, id1 in enumerate(idx1):
        t2 = idx2[id1]

        if i == t2:
            cnt += 1

    print(cnt / len(idx1))
    # one_arr.append(cnt / len(idx1))


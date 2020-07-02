import torch
import numpy as np
import scipy.sparse.linalg as lg
from ..utils import *
from .models import *

def GetProbTranMat(Ak, node_size):
    probTranMat = torch.log(Ak/torch.Tensor.repeat(
        Ak.sum(0), (node_size, 1))) \
        - torch.log(torch.scalar_tensor(1.0 / node_size))
    probTranMat[probTranMat < 0] = 0
    probTranMat[torch.isnan(probTranMat)] = 0
    return probTranMat


class GraRep(ModelWithEmbeddings):
    def __init__(self, kstep, dim, **kwargs):
        super(GraRep, self).__init__(kstep=kstep, dim=int(dim/kstep), **kwargs)

    @classmethod
    def check_train_parameters(cls, **kwargs):
        check_existance(kwargs, {'kstep': 4, 'dim': 128})
        check_range(kwargs, {"kstep": 'positive', 'dim': 'positive'})
        assert kwargs['dim'] % kwargs['kstep'] == 0
        return kwargs

    def GetRepUseSVD(self, probTranMat, alpha):
        U, S, VT = lg.svds(probTranMat, k=self.dim)
        Ud = torch.from_numpy(U)
        Sd = torch.from_numpy(S)
        return torch.as_tensor(Ud)*torch.pow(Sd, alpha)

    def get_train(self, graph, **kwargs):
        adj = torch.from_numpy(graph.adjmat(directed=False, weighted=False, scaled=1)).type(torch.float32)
        Ak = torch.eye(graph.nodesize)
        RepMat = torch.zeros((graph.nodesize, int(self.dim * self.kstep)))
        for i in range(self.kstep):
            print('kstep =', i)
            Ak = torch.mm(Ak, adj)
            probTranMat = GetProbTranMat(Ak, graph.nodesize)
            Rk = self.GetRepUseSVD(probTranMat, 0.5)
            Rk = torch.nn.functional.normalize(Rk, p=2, dim=1)
            RepMat[:, self.dim*i:self.dim*(i+1)] = Rk
        return RepMat

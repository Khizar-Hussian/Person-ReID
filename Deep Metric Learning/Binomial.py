import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


def cosine_similarity(x, y):
    """
        input:
        xi is a vector of n feature vectors each of size 500
        xj is a vector of m feature vectors each of size 500

        output:
        S is a nxm similarity matrix based on cosine similarity function
    """
    S = Variable(torch.zeros(len(x), len(y)), requires_grad=True)
    for i in range(len(x)):
        for j in range(len(y)):
            if j>=i:
                numerator = torch.dot(x[i].t(), y[j])
                denominator = torch.sqrt(torch.mul(torch.dot(x[i].t(), x[i]), torch.dot(y[j].t(), y[j])))
                S[i][j] = torch.div(numerator, denominator)
    return S

class BinomialDevianceLoss(nn.Module):
    def __init__(self, alpha=2, beta=0.5, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, y, m, w):
        # computer similarity matrix
        # s = cosine_similarity(x, y).cuda()
        S = Variable(torch.zeros(len(x), len(y)), requires_grad=True).cuda()
        for i in range(len(x)):
            for j in range(len(y)):
                if j>=i:
                    numerator = torch.dot(x[i].t(), y[j])
                    denominator = torch.sqrt(torch.mul(torch.dot(x[i].t(), x[i]), torch.dot(y[j].t(), y[j])))
                    S[i][j] = torch.div(numerator, denominator)
        m = Variable(m, requires_grad=True).cuda()
        w = Variable(w, requires_grad=True).cuda()

        loss = torch.mul(w, torch.log(1 + torch.exp(torch.mul(-self.alpha*(S-self.beta), m)))).sum()
        # loss = torch.sum(loss)
        return loss
import torch
import torch.nn as nn
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt


class GMM(nn.Module):
    def __init__(self, num_components, mu, sigma):
        super(GMM, self).__init__()
        self.num_components=num_components
        self.mu=nn.Parameter(mu, requires_grad=True)
        self.sigma=nn.Parameter(sigma, requires_grad=True)

    def forward(self, x):
        pdfs=torch.exp(-0.5*((x[:,None,:]-self.mu)/self.sigma)**2)/(self.sigma*np.sqrt(2*np.pi))
        weights=torch.ones(self.num_components)/self.num_components
        pdfs_weighted=(pdfs*weights).sum(dim=1)
        pdfs_norm=pdfs_weighted/pdfs_weighted.sum(dim=1, keepdim=True)
        samples=torch.multinomial(pdfs_norm, num_samples=1)
        return self.mu[samples][:,0,:]

# torch.manual_seed(42)
#
# num_components=2
# mu=torch.randn(num_components, 2)
# sigma=torch.ones(num_components, 2)
#
# gmm=GMM(num_components, mu, sigma)
#
# num_samples=1000
# samples=gmm(torch.empty(num_samples,2).normal_())
# sample_x=samples[:,0].detach().numpy()
# sample_y=samples[:,1].detach().numpy()
# X=[]
# for x in sample_x:
#     X.append(x)
# Y=[]
# for y in sample_y:
#     Y.append(y)

# print(len(X))
# print(Y)
# plt.scatter(samples[:, 0], samples[:, 1])
# plt.show(samples[:, 0], samples[:, 1])
# plt.plot(samples[:, 0],samples[:, 1])
# plt.scatter(X,Y)
#
#
# plt.show()




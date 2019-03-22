# Markov
import numpy as np

matrix = np.matrix([[0.9, 0.075, 0.025], [0.15, 0.8, 0.05], [0.25, 0.25, 0.5]], dtype=float)
v1 = np.matrix([0.05, 0.75, 0.2], dtype=float)
for i in range(100):
    matrix = matrix * matrix
    # print(matrix,i)

import random
from scipy.stats import norm
import matplotlib.pyplot as plt


# M-H
def norm_dist_prob(theta):
    # pdf: probability density function, loc - location parameter
    y = norm.pdf(theta, loc=3, scale=2)
    return y


T = 5 # should be larger
pi = [0 for _ in range(T)]
sigma = 1
t = 0
while t < T - 1:
    t = t + 1
    # p is the sample collection element from pi
    # rvs: random variable of given type -> 条件概率分布Q中采样p
    # 从t-1位子随机生成的下一位p_star
    pi_star = norm.rvs(loc=pi[t - 1], scale=sigma, size=1, random_state=None)
    # M-H Q对称下的上界
    alpha = min(1, (norm_dist_prob(pi_star[0]) / norm_dist_prob(pi[t - 1])))
    u = random.uniform(0, 1)

    if u < alpha:
        pi[t] = pi_star[0]
    else:
        # 这里有些许不同，标准算法中是令t = max(t-1,0)
        t = t -1

# 可视化采样集
# 仿真出pdf loc 3 scale 2
weights = np.ones_like(pi)/float(len(pi))
#plt.scatter(pi, norm.pdf(pi, loc=3, scale=2))
num_bins = 50
#plt.hist(pi, num_bins, density=True, facecolor='red', alpha=0.6,weights=np.ones_like(pi)/len(pi))
#plt.show()

# 2 dimensional gibbs
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import math
sample_source = multivariate_normal(mean=[5,-1],cov=[[1,.5],[.5,2]])

rho = .5

def p_ygivenx(x,m1,m2,s1,s2):
    return random.normalvariate(m2+ rho*s2/s1 * (x-m1),math.sqrt(1-rho**2)*s2)


def p_xgiveny(y,m1,m2,s1,s2):
    return random.normalvariate(m1+rho*s1/s2*(y-m2),math.sqrt(1-rho**2)*s1)

N = 5000
K = 20
x_res = []
y_res = []
z_res = []
m1 = 5
m2 = -1
s1 = 1
s2 = 2

y = m2

for i in range(N):
    for j in range(K):
        x = p_xgiveny(y,m1,m2,s1,s2)
        y = p_ygivenx(x,m1,m2,s1,s2)
        z = sample_source.pdf([x,y])
        x_res.append(x)
        y_res.append(y)
        z_res.append(z)

num_bins = 50
plt.hist(x_res,num_bins,normed = 1,facecolor='green',alpha = .5)
plt.hist(y_res,num_bins,normed = 1,facecolor='red',alpha = .5)
plt.show()

fig = plt.figure()
ax = Axes3D(fig,rect=[0,0,1,1],elev=30,azim=20)
ax.scatter(x_res,y_res,z_res,marker='o')
plt.show()

















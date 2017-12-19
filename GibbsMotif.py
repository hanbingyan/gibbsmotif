import numpy as np
from scipy import stats
from scipy.special import factorial
import re
import time

start = time.time()
with open('D:/GoogleDrive/Sync/5050hw2/hw2_part2_data.txt', 'r') as content_file:
    content = content_file.read().replace('\n', '')

content = re.sub('\d+', '', content)
content = content.upper()
data = np.array(re.split('>', content)[1:])

# data = np.array(['TAAAT', 'CAAAG', 'AAAGT', 'CGAAA', 'TCAAA', 'AAACT', 'GAAAG'])
J = 18  # len of motif
L = len(data[1])  # len of each DNA string
K = data.shape[0]  # total number of seq
p = 4  # DNA A0 C1 G2 T3
dic = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
theta = np.ones(p) / p
Theta = np.ones((p, J)) / p
A = np.ones(K, dtype=np.int)
alpha = np.ones(p) / 2
B = np.ones((p, J))
postA = np.zeros((100, K))
posttheta = np.zeros((100, p))
for it in range(400):
    Aprob = np.zeros((K, L - J + 1))
    for k in range(K):
        for l in range(L - J + 1):
            A[k] = l
            h = np.array((-data[k][l:l + J].count('A'), -data[k][l:l + J].count('C'),
                          -data[k][l:l + J].count('G'), -data[k][l:l + J].count('T')))

            pi = 1  # pi is \Pi^J_{j=1} h...
            for j in range(J):
                ch = data[k][l + j]
                temp = [data[idx][A[idx] + j] for idx in range(K)].count(ch)
                pi *= temp - 1 + B[dic[ch], j]
            Aprob[k, l] = np.prod(np.power(theta, h)) * pi  # non-normalized probab a_k = l
        A[k] = np.random.choice(L - J + 1, size=1, p=Aprob[k, :] / np.sum(Aprob[k, :]))
    hRAc = np.zeros(p)
    for nucacid in ['A', 'C', 'G', 'T']:
        for idx in range(K):
            hRAc[dic[nucacid]] += data[idx].count(nucacid) - data[idx][A[idx]:A[idx] + J].count(nucacid)
    theta = np.random.dirichlet(hRAc + alpha, size=1)
    if it % 10 == 0:
        delta = 0
        if np.max(A) == L - J or np.min(A) == 0:
            MHprob = 0
        else:
            delta = np.random.randint(2, size=1)
            if delta == 1:
                strhAplusJ = [data[idx][A[idx] + J] for idx in range(K)]
                hAplusJ = [strhAplusJ.count(nucacid) for nucacid in ['A', 'C', 'G', 'T']]
                strhA = [data[idx][A[idx]] for idx in range(K)]
                hA = [strhA.count(nucacid) for nucacid in ['A', 'C', 'G', 'T']]
                MHprob = np.prod(factorial(hAplusJ)) * np.prod(np.power(theta, hA)) / np.prod(factorial(hA)) / np.prod(
                    np.power(theta, hAplusJ))
            else:
                delta = -1
                strhAplusJm1 = [data[idx][A[idx] + J - 1] for idx in range(K)]
                hAplusJm1 = [strhAplusJm1.count(nucacid) for nucacid in ['A', 'C', 'G', 'T']]
                strhAm1 = [data[idx][A[idx] - 1] for idx in range(K)]
                hAm1 = [strhAm1.count(nucacid) for nucacid in ['A', 'C', 'G', 'T']]
                MHprob = np.prod(factorial(hAm1)) * np.prod(np.power(theta, hAplusJm1)) / np.prod(
                    factorial(hAplusJm1)) / np.prod(np.power(theta, hAm1))
        if np.random.uniform(0, 1, size=1) < MHprob:
            A = A + delta
    if it % 40 == 0:
        print('Iteration %d ...' % (it))
    if it > 299:
        postA[it - 300, :] = A
        posttheta[it - 300, :] = theta
modeA, countA = stats.mode(postA)
modeA = modeA.ravel()
modeA = modeA.astype(int)
print('--- Posterior mode of A --- \n')
print(modeA)
f = open('D:/Seq.txt', 'w')
for k in range(K):
    f.write(data[k][modeA[k]:modeA[k] + J])
    f.write('\n')
f.close()
print('--- Posterior mean of background, theta --- \n')
print(np.mean(posttheta, axis=0))
for j in range(J):
    hstr = [data[idx][modeA[idx] + j] for idx in range(K)]
    Theta[:, j] = np.array([hstr.count(nucacid) for nucacid in ['A', 'C', 'G', 'T']]) / K
print('--- Probability matrix of motif, Theta --- \n')
print('A     C     G     T\n')
print(Theta.T)
print('It took', time.time() - start, 'seconds in total.')

import numpy as np
import faiss

d = 64
# 向量维度
nb = 100000
# 待索引向量size
nq = 10000
# 查询向量size
np.random.seed(1234)
# 随机种子确定
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000
#为了使随机产生的向量有较大区别进行人工调整向量
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000
print(1)

import time

start = time.time()
index = faiss.IndexFlatL2(d)
# 建立索引
print(index.is_trained)        
# 输出true
index.add(xb)
# 索引中添加向量
print(index.ntotal)            
# 输出100000
k = 4
# 返回每个查询向量的近邻个数
D, I = index.search(xb[:5], k)
# 检索check
print(I)
print(D)
D, I = index.search(xq, k)
#xq检索结果
print(I[:5])
# 前五个检索结果展示
print(I[-5:])
# 最后五个检索结果展示
end1 = time.time()


nlist = 100
k = 4
quantizer = faiss.IndexFlatL2(d)
# 量化器索引
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
# 指定用L2距离进行搜索，若不指定默认为內积
assert not index.is_trained
index.train(xb)   
# 索引训练
assert index.is_trained
index.add(xb)
# 向量添加
D, I = index.search(xq, k)
# 检索
print(I[-5:])
# 最后五个检索结果
index.nprobe = 10
# 多探针检索
D, I = index.search(xq, k)
print(I[-5:])
#最后五个检索结果
end2 = time.time()
print('method1:{}s'.format(end1-start))
print('method2:{}s'.format(end2-end1))

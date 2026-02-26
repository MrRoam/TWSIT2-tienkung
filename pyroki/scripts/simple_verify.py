"""
最简单的验证：reshape矩阵乘法的正确性
"""
import numpy as np

R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

# 2个向量
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

print("方法1：逐个变换")
r1 = R @ v1
r2 = R @ v2
print("r1 =", r1)
print("r2 =", r2)

print("\n方法2：批量reshape")
batch = np.array([v1, v2])  # (2, 3)
batch_T = batch.T  # (3, 2)
result_T = R @ batch_T  # (3, 3) @ (3, 2) = (3, 2)
result = result_T.T  # (2, 3)

print("批量结果:")
print(result[0])
print(result[1])

print("\n验证：")
print("r1 == result[0]?", np.allclose(r1, result[0]))
print("r2 == result[1]?", np.allclose(r2, result[1]))

print("\n结论：")
print("reshape后矩阵乘法 = 逐个向量变换")
print("因为：R @ [v1, v2] = [R@v1, R@v2]")

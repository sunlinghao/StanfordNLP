import nltk
import numpy as np
# longest common subsequence


s1='qwertyuiop'
s2='qawsedrftgyhujilp'

l1 = len(s1)
l2 = len(s2)

arr1 = [[0] * (l2 + 1) for i_arr in range(l1+1)]
arr1 = np.array(arr1)
print(arr1)
for i in range(1, l1+1):
    for j in range(1, l2+1):
        if s1[i-1] == s2[j-1]:
            arr1[i,j] = arr1[i-1, j-1] + 1
        elif arr1[i-1, j] >= arr1[i, j-1]:
            arr1[i, j] = arr1[i-1, j]
        else:
            arr1[i,j] = arr1[i, j-1]

print(arr1)

def print_LCS(X,i,j):
    if i == 0 and j == 0:
        return
    if X[i,j]>X[i-1,j] and X[i,j]>X[i,j-1] and X[i,j]>X[i-1,j-1]:
        print_LCS(X,i-1,j-1)
        print(s2[j-1])
    elif X[i,j] == X[i,j-1]:
        print_LCS(X,i,j-1)
    elif X[i,j] == X[i-1,j]:
        print_LCS(X,i-1,j)

print_LCS(arr1,10,17)

ls = []
ls1 = []
ls2 = ls + ls1
s = "ejshfkadsljfh"
print(s[0])

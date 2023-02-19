from dataloader import data_loader

# available cities
# madrid, munchen, minneapolis, saintpaul, richmond, littlerock, portland, washington, paris, firenze, atlanta, frankfurtammain

A, x1, x2, x3, x4, x5 = data_loader("firenze")

# type(A) : <class 'scipy.sparse._lil.lil_matrix'>
# type(x1) : <class 'numpy.ndarray'>
# type(x1[i]) : <class 'numpy.ndarray'>

"""
example of x3 
[[11.2292767 43.7892378  0.         0.         0.       ] -> x, y, 0, 0, 0
  [ 4.         0.         1.         0.         0.       ]]] -> one-hot encoding (motorway+trunk+primary/+secondary/+tertiary/+unclassified/+residential)
"""

print(A)
# print(type(A))
print(x3)


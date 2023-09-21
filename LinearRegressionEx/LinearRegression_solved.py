
import numpy as np
import matplotlib.pyplot as plt

# generate random data-set
# np.random.seed(0) # choose random seed (optional)
x = np.random.rand(100, 1)
y = 2 + 3 * x + np.random.rand(100, 1)

# J = 0 # initialize J, this can be deleted once J is defined in the loop
w = np.matrix([np.random.rand(),np.random.rand()]) # slope and y-intercept
a = 0.001 # learning rate step size
ite = 100 # number of training iterations

jList = []
numIte = []

# Write Linear Regression Code to Solve for w (slope and y-intercept) Here ##
for p in range (ite):
    for i in range(len(x)):
        # Calculate w and J here
        x_vec = np.matrix([x[i][0],1]) # Option 1 | Setting up a vector for x (x_vec[j] corresponds to w[j])
        h = w * x_vec.T ## Hint: you may need to transpose x or w by adding .T to the end of the variable
        w = w - a * (h - y[i]) * x_vec
        J = (1/2) * (((h - y[i])) ** 2)
        J = J.item()
        
    jList.append(J)
    numIte.append(p)
    print('Loss:', J)

## if done correctly the line should be in line with the data points ##

print('f = ', w[0,0],'x + ', w[0,1])
# print(jList)
# print(np.shape(jList))

# plot
plt.figure(1)
plt.scatter(x,y,s=ite)
plt.plot(x, w[0,1] + (w[0,0] * x), linestyle='solid')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure(2)
plt.plot(jList)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

import numpy as np
import matplotlib.pyplot as plt


def load_2d_sample_data(file):
    f = open(file, 'r')
    line_0 = f.readline().split()
    line_1 = f.readline().split()
    return np.array([line_0, line_1]).astype('float')

# ================================================================
# Step 0: Load data
#  We have provided the code to load data from pcaData.txt into x.
#  x is a 2 * 45 matrix, where the kth column x(:,k) corresponds to
#  the kth data point.Here we provide the code to load natural image data into x.
#  You do not need to change the code below.

x = load_2d_sample_data('data/pcaData.txt')
f1 = plt.figure(1)
plt.scatter(x[0], x[1])
f1.suptitle('Raw data')
f1.show()

# ================================================================
# Step 1a: Implement PCA to obtain U
#  Implement PCA to obtain the rotation matrix U, which is the eigenbasis
#  sigma.

# -------------------- YOUR CODE HERE --------------------
u = np.zeros([len(x), len(x)])  # You need to compute this





# --------------------------------------------------------
f2 = plt.figure(2)
plt.plot([0.0, u[0][0]], [0.0, u[1][0]], color='b')
plt.plot([0.0, u[0][1]], [0.0, u[1][1]], color='b')
plt.scatter(x[0], x[1])
plt.xlim([-1.0, 1.0])
plt.ylim([-1.0, 1.0])
f2.suptitle('Raw data with u vectors')
f2.show()

# ================================================================
# Step 1b: Compute xRot, the projection on to the eigenbasis
#  Now, compute xRot by projecting the data on to the basis defined
#  by U. Visualize the points by performing a scatter plot.

# -------------------- YOUR CODE HERE --------------------
x_rot = np.zeros(x.shape)





# --------------------------------------------------------
# Visualise the covariance matrix. You should see a line across the
# diagonal against a blue background.
f3 = plt.figure(3)
plt.scatter(x_rot[0], x_rot[1])
f3.suptitle('x_rot')
f3.show()

# ================================================================
# Step 2: Reduce the number of dimensions from 2 to 1.
#  Compute xRot again (this time projecting to 1 dimension).
#  Then, compute xHat by projecting the xRot back onto the original axes 
#  to see the effect of dimension reduction

# -------------------- YOUR CODE HERE -------------------- 
k = 1  # Use k = 1 and project the data onto the first eigenbasis
x_hat = np.zeros(x.shape)





# --------------------------------------------------------
f4 = plt.figure(4)
plt.scatter(x_hat[0], x_hat[1])
f4.suptitle('x_hat')
f4.show()

# ================================================================
# Step 3: PCA Whitening
#  Complute xPCAWhite and plot the results.

epsilon = 1e-5
# -------------------- YOUR CODE HERE -------------------- 
x_pca_white = np.zeros(x.shape)  # You need to compute this




# --------------------------------------------------------
f5 = plt.figure(5)
plt.scatter(x_pca_white[0], x_pca_white[1])
f5.suptitle('x_pca_white')
f5.show()

# ================================================================
# Step 3: ZCA Whitening
#  Complute xZCAWhite and plot the results.

# -------------------- YOUR CODE HERE -------------------- 
# xZCAWhite = zeros(size(x)); # You need to compute this
x_zca_white = np.zeros(x.shape)




# --------------------------------------------------------
f6 = plt.figure(6)
plt.scatter(x_zca_white[0], x_zca_white[1])
f6.suptitle('x_zca_white')
f6.show()

# Congratulations! When you have reached this point, you are done!
#  You can now move onto the next PCA exercise. :)
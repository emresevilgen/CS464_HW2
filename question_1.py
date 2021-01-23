from numpy.linalg import svd
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def loadImages(path):
    # Put files into lists and return them as one list
    image_files = []
    for file in os.listdir(path):
        if file.endswith('.jpg'):
            image_files.append(os.path.join(path, file))

    return image_files


def display_img(image, path):
    plt.title(path)
    plt.imshow(image)
    plt.show()


def preprocess(path):
    # Generate image file paths
    filepaths = loadImages(path)

    first = True

    for path in filepaths:
        # loading image
        image = mpimg.imread(path)

        # if an image is grayscale, stack the 64x64 image along the
        # third dimension to obtain RGB version of size 64x64x3
        if len(image.shape) == 2:
            image = np.stack((image, image, image), axis=2)

        # Flatten all images of size 64x64x3 to get 4096x3 matrix for each image
        image = np.reshape(image, (image.shape[0]*image.shape[1], 3))

        if first:
            # Create a 3-D array, X, of size Dx4096x3 by stacking flattened
            # matrices of the images provided in the dataset.
            X = np.expand_dims(image, axis=0)
            first = False
        else:
            image = np.expand_dims(image, axis=0)
            X = np.vstack((X, image))

    # Slice X as Xi = X[:; :; i] where i corresponds to the first
    # three index, for obtaining each color channel matrix independently
    # for all images. Reshape all Xi to obtain matrices, instead of 3D arrays
    X0 = X[:, :, 0]
    X1 = X[:, :, 1]
    X2 = X[:, :, 2]

    return X0, X1, X2


# Question 1.1
# Preprocess and get the datasets for each color
image_path = os.path.join(".", "datasets", "q1_dataset", "van_gogh")
X0, X1, X2 = preprocess(image_path)

# Obtain first 100 principal components per color channel matrix for each dataset
u0, s0, Vt0 = svd(X0)
u1, s1, Vt1 = svd(X1)
u2, s2, Vt2 = svd(X2)

# Sort the first 100 singular values in descending order and plot a bar chart per Xi
pcs0 = s0[:100]
pcs1 = s1[:100]
pcs2 = s2[:100]

# Bar chart for singular values of X0
plt.figure()

y_pos = np.arange(len(pcs0))
plt.bar(y_pos, pcs0)
plt.title('Singular Values of X0')

plt.xlabel('Principal Components')
plt.ylabel('Singular Values')

# Bar chart for singular values of X1
plt.figure()

y_pos = np.arange(len(pcs1))
plt.bar(y_pos, pcs1)
plt.title('Singular Values of X1')

plt.xlabel('Principal Components')
plt.ylabel('Singular Values')

# Bar chart for singular values of X2
plt.figure()

y_pos = np.arange(len(pcs2))
plt.bar(y_pos, pcs2)
plt.title('Singular Values of X2')

plt.xlabel('Principal Components')
plt.ylabel('Singular Values')

plt.draw()

# Report proportion of variance explained (PVE) by the first 10 principal components
pve0 = sum(map(lambda i: i * i, pcs0[:10])) / sum(map(lambda i: i * i, pcs0))
pve1 = sum(map(lambda i: i * i, pcs1[:10])) / sum(map(lambda i: i * i, pcs1))
pve2 = sum(map(lambda i: i * i, pcs2[:10])) / sum(map(lambda i: i * i, pcs2))

print("PVE for X0:", pve0)
print("PVE for X1:", pve1)
print("PVE for X2:", pve2)

# Question 1.2


def get_images(path):
    # Generate image file paths
    filepaths = loadImages(path)

    first = True

    for path in filepaths:
        # loading image
        image = mpimg.imread(path)

        # if an image is grayscale, stack the 64x64 image along the
        # third dimension to obtain RGB version of size 64x64x3
        if len(image.shape) == 2:
            image = np.stack((image, image, image), axis=2)

        if first:
            # Create a 3-D array, X, of size Dx4096x3 by stacking flattened
            # matrices of the images provided in the dataset.
            X = np.expand_dims(image, axis=0)
            first = False
        else:
            image = np.expand_dims(image, axis=0)
            X = np.vstack((X, image))

    return X


image_path = os.path.join(".", "datasets", "q1_dataset", "van_gogh")
X = get_images(image_path)

# Compute the mean and variance of the dataset
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

# Add noise to each image by sampling from a 64x64x3 dimensional gaussian with mean and variance
gaussian = np.random.normal(mean, std, mean.shape)

# Scale the noise with a factor of 0:01 before they are added to the images
gaussian = gaussian / 100

for i in range(X.shape[0]):
    X[i] = X[i] + gaussian

X = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2], X.shape[3]))

# Slice X as Xi = X[:; :; i] where i corresponds to the first
# three index, for obtaining each color channel matrix independently
# for all images. Reshape all Xi to obtain matrices, instead of 3D arrays
X0 = X[:, :, 0]
X1 = X[:, :, 1]
X2 = X[:, :, 2]

# Obtain first 100 principal components per color channel matrix for each dataset
u0, s0, Vt0 = svd(X0)
u1, s1, Vt1 = svd(X1)
u2, s2, Vt2 = svd(X2)

# Sort the first 100 singular values in descending order and plot a bar chart per Xi
pcs0 = s0[:100]
pcs1 = s1[:100]
pcs2 = s2[:100]

# Bar chart for singular values of X0 with Noise
plt.figure()

y_pos = np.arange(len(pcs0))
plt.bar(y_pos, pcs0)
plt.title('Singular Values of X0 with Noise')

plt.xlabel('Principal Components')
plt.ylabel('Singular Values')

# Bar chart for singular values of X1 with Noise
plt.figure()

y_pos = np.arange(len(pcs1))
plt.bar(y_pos, pcs1)
plt.title('Singular Values of X1 with Noise')

plt.xlabel('Principal Components')
plt.ylabel('Singular Values')

# Bar chart for singular values of X2 with Noise
plt.figure()

y_pos = np.arange(len(pcs2))
plt.bar(y_pos, pcs2)
plt.title('Singular Values of X2 with Noise')

plt.xlabel('Principal Components')
plt.ylabel('Singular Values')

plt.show()

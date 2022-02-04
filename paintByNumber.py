from PIL import Image, ImageChops, ImageFilter, ImageMath
from PIL.ImageFilter import CONTOUR
from numpy import asarray
import numpy as np
from sklearn import cluster
import sklearn as sk


# Helper function to ensure valid argument
def check_for_posint(string):
    try:
        int(string)
        if int(string) <= 0:
            return False
        else:
            return True
    except ValueError:
        return False


# Helper function to create transparent white pixels
def transparent_white(img):
    grid = img.convert("RGBA")
    rec_data = grid.getdata()
    new_data = []
    for i in rec_data:
        if i[0] == 255 and i[1] == 255 and i[2] == 255:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(i)
    grid.putdata(new_data)

    return grid


image = Image.open('C:\\image.jpg')

# image.show()

# Convert image to numpy array
image = asarray(image)

# Reshape the array to 2-D: (pixel x RGB)
data = image.reshape(-1, 3)

resolutions = input("Please enter a valid number of colors in the output: ")

while not check_for_posint(resolutions):
    input("Please enter a valid number of colors in the output: ")

resolutions = int(resolutions)

# K-means clustering
kmeans = sk.cluster.KMeans(n_clusters=resolutions)

kmeans.fit(data)

segmented_images = kmeans.cluster_centers_[kmeans.labels_].reshape(image.shape)

# Image given from the K-means method
img_out = Image.fromarray(segmented_images.astype(np.uint8))

# PIL contour function to find edges
img_contour = img_out.filter(CONTOUR)

# Convert anything above threshold to black, anything else to white
threshold = 200
mapper = lambda x: 255 if x > threshold else 0
sketch = img_contour.convert('L').point(mapper, mode='1')

# Black and white "sketch" of original image
sketch.show()
sketch.save("C:\\sketch.jpg")

my_grid = transparent_white(sketch)

# Results from K-clustering
img_out.show()
img_out.save("C:\\gridless.jpg")

# Combine K-clustering results and "sketch" to give what a potential fill-in could look like
img_out.paste(my_grid, (0, 0), my_grid)

# Result from combining results and "sketch"
img_out.show()
img_out.save("C:\\final.jpg")


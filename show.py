from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
file = '/Users/jihan/Desktop/visualizations_validcrf00000001.jpg'
img = Image.open(file)
img = img.resize((256, 256))
img_arr = np.array(img)
rows, cols = img_arr.shape
img_arr[img_arr == 1] = 1
plt.imshow(img_arr)
plt.show()
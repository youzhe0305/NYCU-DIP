import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import time

def conv(image, kernel): # assume kernel is odd size
    n, m = kernel.shape
    kernel = np.flip(kernel, axis=(0, 1)) # flip up-down and left-right => convolution (shape m,n)
    image = np.pad(image, ((n//2, n//2), (m//2, m//2)), mode='constant', constant_values=0) # zero-padding, same convolution
    for i in range(image.shape[0]-m+1):
        for j in range(image.shape[1]-n+1):
            image[i][j] = np.clip(np.sum(image[i:i+m, j:j+n] * kernel), 0, 255)
    return image[:-n+1, :-m+1] # remove padding

def Laplacian_spatial(image):
    print('image shape:', image.shape)
    kernel_1 = np.array([[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]])
    kernel_2 = np.array([[-1, -1, -1],
                         [-1, 9, -1],
                         [-1, -1, -1]])
    img1 = conv(image, kernel_1)
    img2 = conv(image, kernel_2)
    return img1, img2



def save_image(image, title):
    cv2.imwrite(title +  '.jpg', image)

def downsample(image, factor):
    return zoom(image, factor, order=3) # bicubic interpolation

def main():
    image = cv2.imread('./altar.jpg', 0) # read image as grayscale
    if type(image) == type(None):
        print('Error: image not found')
        return
    image = downsample(image, 1/4)
    
    save_image(image, 'ori_img_spatial')    
    start = time.time()
    img1, img2 = Laplacian_spatial(image)
    end = time.time()
    print(f"Laplacian_spatial Time: {end - start:.6f} seconds")

    save_image(img2, 'Lap_spatial_2')
    save_image(img1, 'Lap_spatial_1')

if __name__ == "__main__":
    main()
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import time

def Laplacian_frequency(image):
    
    f_transform = np.fft.fft2(image)
    # f_transform_shifted = f_transform
    f_transform_shifted = np.fft.fftshift(f_transform) # shift zero frequency to center
    
    # construct distance map
    u = np.arange(0, image.shape[0])
    v = np.arange(0, image.shape[1])
    u, v = np.meshgrid(u, v, indexing='ij')
    u0 = image.shape[0] // 2
    v0 = image.shape[1] // 2
    dis = np.sqrt((u - u0)**2 + (v - v0)**2)
    
    H_ = (4 * (np.pi**2) * (dis**2)) # - H
    H_normalized = (H_ - H_.min()) / (H_.max() - H_.min()) # normalize to weight [0,1]
    factor = 3
    Lap_transform = f_transform_shifted * (1 + factor * H_normalized) # high frequency will get higher weight

    image_ret = np.fft.ifft2(np.fft.ifftshift(Lap_transform)).real
    image_ret = np.clip(image_ret, 0, 255).astype(np.uint8)
    return image_ret

    
def save_image(image, title):
    cv2.imwrite(title + '.jpg', image)

def downsample(image, factor):
    return zoom(image, factor, order=3) # bicubic interpolation

def main():
    image = cv2.imread('./altar.jpg', 0) # read image as grayscale
    if type(image) == type(None):
        print('Error: image not found')
        return
    image = downsample(image, 1/4)
    save_image(image, 'ori_img_frequency')
    
    start = time.time()
    image_ret = Laplacian_frequency(image)
    end = time.time()
    print(f"Laplacian_frequency Time: {end - start:.6f} seconds")

    save_image(image_ret, 'Lap_frequency_1')


if __name__ == "__main__":
    main()
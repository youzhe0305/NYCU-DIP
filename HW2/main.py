import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_hist(img, n):
    interval = 256 / n
    levels = (img / interval).astype(np.uint8)
    hist = np.bincount(levels.flatten(), minlength=n).astype(np.float32)
    hist = hist / (img.shape[0] * img.shape[1])# hist.sum() # make hist probability
    return hist

def hist_equalization(img, hist, n):
    s = np.cumsum(hist)
    s = s * (n - 1)
    s = (s + 0.5).astype(np.uint8)
    interval = 256 / n
    levels = (img / interval).astype(np.uint8)
    img = img + (s[levels] - levels.astype(np.float32)) * interval
    return img.astype(np.uint8)

def hist_specification(img1, img2, n): # img1 -> img2
    hist1 = generate_hist(img1, n)
    s = np.cumsum(hist1)
    s = s * (n - 1)
    s = (s + 0.5).astype(np.uint8)
    hist2 = generate_hist(img2, n)
    z = np.cumsum(hist2)
    z = z * (n - 1)
    z = (z + 0.5).astype(np.uint8)
    inv_z = np.zeros(n)
    for i in range(0, n):
        idx = np.where(z == i)[0] # fine first i in z
        idx = idx[0] if idx.size > 0 else -1
        inv_z[i] = idx
    canva = np.zeros(img1.shape)
    interval = 256 / n
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            level = (img1[i][j] / interval).astype(np.uint8)
            to_level = s[level]
            while inv_z[to_level] == -1:
                to_level -= 1
            to_level = inv_z[to_level].astype(np.float32)
            canva[i][j] = img1[i][j] + (to_level - level.astype(np.float32)) * interval
    return canva.astype(np.uint8)
    
if __name__ == '__main__':
    
    n = 256 # number of histogram intervals
    img = cv2.imread('./Q1.jpeg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img1 = cv2.imread('./Q2_source.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img2 = cv2.imread('./Q2_reference.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    hist = generate_hist(img, n)
    # print(hist)
    img_equal = hist_equalization(img, hist, n)
    cv2.imwrite('Q1_ans.jpg', img_equal)
    img_spec = hist_specification(img1, img2, n)
    cv2.imwrite('Q2_ans.jpg', img_spec)

    plt.bar(range(256), hist)
    plt.title("hist_ori")
    plt.show()

    hist_equal = generate_hist(img_equal, n)
    plt.bar(range(256), hist_equal)
    plt.title("hist_equal")
    plt.show()

    hist_source = generate_hist(img1, n)
    plt.bar(range(256), hist_source)
    plt.title("hist_source")
    plt.show()  

    hist_ref = generate_hist(img2, n)
    plt.bar(range(256), hist_ref)
    plt.title("hist_ref")
    plt.show()    
    
    hist_spec = generate_hist(img_spec, n)
    plt.bar(range(256), hist_spec)
    plt.title("hist_spec")
    plt.show()



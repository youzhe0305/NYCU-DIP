import cv2
import numpy as np
import time

def compute_H(pts1, pts2): # correspond points in img1 & img2

    '''
    [wx']       [x]
    [wy'] = H * [y]
    [ w ]       [1]
    
    w = x*h_31 + y*h_32 + h_33
    x' = (x*h_11 + y*h_12 + h_13) / (x*h_31 + y*h_32 + h_33)
    y' = (x*h_21 + y*h_22 + h_33) / (x*h_31 + y*h_32 + h_33)
    
    x*h_11 + y*h_12 + h_13 - x'*x*h_31 - x'*y*h_32 - x'*h_33 = 0
    x*h_11 + y*h_12 + h_13 - y'*x*h_31 - y'*y*h_32 - y'*h_33 = 0

    h_33 only affect scale of value, set h_33 = 1

    [x, y, 1, 0, 0, 0, -x'x, -x'y]   [h_11]   [x']
    [0, 0, 0, x, y, 1, -y'x, -y'y] * [h_12] = [y']
                                     [h_13]   
                                     [h_21]
                                     ...
    
    HA = B, A,B matrix including points
    '''

    A = []
    B = []
    for pt1, pt2 in zip(pts1, pts2):
        x = pt1[0]
        y = pt1[1]
        xp = pt2[0]
        yp = pt2[1]
        A.append([x, y, 1 ,0, 0, 0, -xp*x, -xp*y])
        A.append([0, 0, 0 ,x, y, 1, -yp*x, -yp*y])
        B.append([xp])
        B.append([yp])
    A = np.array(A)
    B = np.array(B)
    H = np.linalg.lstsq(A, B, rcond=-1)[0]
    H = np.array([[H[0][0], H[1][0], H[2][0]],
                  [H[3][0], H[4][0], H[5][0]],
                  [H[6][0], H[7][0], 1.]])
    return H

def get_all_points(height, width): # get all pixel in (width, height) square return list = [x,y]
    x_axis = np.arange(0,width,1)
    y_axis = np.arange(0,height,1)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis) # x_grid = (height, width), y_grid = (height, width)
    points = np.stack((x_grid, y_grid), axis=-1) # (height, width, 2)
    points = points.reshape(width*height ,2)
    return points

def nearest_neighbor(img, pt):
    x = int(pt[0] + 0.5)
    y = int(pt[1] + 0.5)
    if x < 0 or y < 0 or x >= img.shape[1] or y >= img.shape[0]:
        return np.zeros(3, dtype=img.dtype)
    return img[y][x]


def bilinear_interpolation(img, pt):
    x = pt[0]
    y = pt[1]
    if x <= -1 or y <= -1 or x >= img.shape[1] or y >= img.shape[0]:
        return np.zeros(3, dtype=img.dtype)
    x1 = int(x)
    x2 = int(x)+1
    y1 = int(y)
    y2 = int(y)+1
    left_top_w = (x2 - x) * (y2 - y)
    left_top_val = img[y1, x1] if x1 >= 0 and y1 >= 0 else 0 # zero padding
    right_top_w = (x - x1) * (y2 - y)
    right_top_val = img[y1, x2] if x2 < img.shape[1] and y1 >= 0 else 0
    left_bottom_w = (x2 - x) * (y - y1)
    left_bottom_val = img[y2, x1] if x1 >= 0 and y2 < img.shape[0] else 0
    right_bottom_w = (x - x1) * (y - y1)
    right_bottom_val = img[y2, x2] if x2 < img.shape[1] and y2 < img.shape[0] else 0
    value = left_top_val * left_top_w + right_top_val * right_top_w + left_bottom_val * left_bottom_w + right_bottom_val * right_bottom_w
    return value

def get_img_value(img, pt):
    x = pt[0]
    y = pt[1]
    if x <= -1 or y <= -1 or x >= img.shape[1] or y >= img.shape[0]:
        return np.zeros(3, dtype=img.dtype)
    else:
        return img[y][x]
def bicubic(pts, target_pt, direction='x'): # input 4 points
    x = target_pt[0] - int(target_pt[0])
    y = target_pt[1] - int(target_pt[1])
    # pts store value of 4 points
    a = -1/2 * pts[0] + 3/2 * pts[1] - 3/2 * pts[2] + 1/2 * pts[3]
    b = pts[0] - 5/2 * pts[1] + 2 * pts[2] - 1/2 * pts[3]
    c = -1/2 * pts[0] + 1/2 * pts[2]
    d = pts[1]
    if direction == 'x':
        return a * x**3 + b * x**2 + c * x + d
    elif direction == 'y':
        return a * y**3 + b * y**2 + c * y + d
    else:
        NotImplementedError

def bicubic_interpolation(img, pt):
    x = pt[0]
    y = pt[1]
    if x < -1 or y < -1 or x >= img.shape[1] or y >= img.shape[0]:
        return np.zeros(3)
    x1 = int(x)-1
    x2 = int(x)
    x3 = int(x)+1
    x4 = int(x)+2
    y1 = int(y)-1
    y2 = int(y)
    y3 = int(y)+1
    y4 = int(y)+2
    bicubic_pts = [get_img_value(img, (x1, y1)), get_img_value(img, (x2, y1)), get_img_value(img, (x3, y1)), get_img_value(img, (x4, y1))] # store 4 points value 
    v1 = bicubic(bicubic_pts, (x, y), direction='x')
    # print('v1:', v1)
    # print(bicubic_pts)
    bicubic_pts = [get_img_value(img, (x1, y2)), get_img_value(img, (x2, y2)), get_img_value(img, (x3, y2)), get_img_value(img, (x4, y2))]
    v2 = bicubic(bicubic_pts, (x, y), direction='x')
    bicubic_pts = [get_img_value(img, (x1, y3)), get_img_value(img, (x2, y3)), get_img_value(img, (x3, y3)), get_img_value(img, (x4, y3))]
    v3 = bicubic(bicubic_pts, (x, y), direction='x')
    bicubic_pts = [get_img_value(img, (x1, y4)), get_img_value(img, (x2, y4)), get_img_value(img, (x3, y4)), get_img_value(img, (x4, y4))]
    v4 = bicubic(bicubic_pts, (x, y), direction='x')
    bicubic_pts = [v1, v2, v3, v4]
    ret = bicubic(bicubic_pts, (x, y), direction='y')
    # print(ret)
    return ret
    
def warp_img(img1, img2, pts1, pts2, mode='bilinear'):
    
    H = compute_H(pts1, pts2)
    # use inverse matrix, start with img2's target points and computer origin points
    # avoid mapping to float point
    H = np.linalg.inv(H)

    canva = np.zeros((img2.shape[0], img2.shape[1], 3))
    points = get_all_points(img2.shape[0], img2.shape[1])
    for pts in points:
        ori_pts = (H @ np.append(pts, 1).T) # add 1 for homogenous coordinate
        ori_pts = (ori_pts / ori_pts[2]) # divide by w, make it homogenous
        if mode == 'bilinear':
            value = np.nan_to_num(bilinear_interpolation(img1, ori_pts))
        elif mode == 'bicubic':
            value = np.nan_to_num(bicubic_interpolation(img1, ori_pts))
        elif mode == 'nearest':
            value = np.nan_to_num(nearest_neighbor(img1, ori_pts))
        canva[pts[1], pts[0],:] = value
    canva = np.clip(canva, 0, 255)
    canva = canva.astype(np.uint8)
    mask = canva == 0
    ret = img2 * mask + canva
    # cv2.imshow('window', ret)
    # cv2.waitKey(0)
    cv2.imwrite(f'warp_{mode}.jpg', ret)

def rotate_img(img, angle=0, mode='bilinear'):
    angle = np.deg2rad(angle)
    cx = img.shape[1] // 2
    cy = img.shape[0] // 2
    rotate = np.array([[np.cos(angle), -np.sin(angle), cx - cx * np.cos(angle) + cy * np.sin(angle)],
                       [np.sin(angle), np.cos(angle), cy - cx * np.sin(angle) - cy * np.cos(angle)],
                       [0, 0, 1]]) # rotate according to the center points
    rotate_inv = np.linalg.inv(rotate)
    canva = np.zeros((img.shape[0], img.shape[1], 3))
    points = get_all_points(img.shape[0], img.shape[1])
    for pts in points:
        ori_pts = (rotate_inv @ np.append(pts, 1).T) # add 1 for homogenous coordinate
        ori_pts = (ori_pts / ori_pts[2]) # divide by w, make it homogenous
        if mode == 'bilinear':
            value = np.nan_to_num(bilinear_interpolation(img, ori_pts))
        elif mode == 'bicubic':
            value = np.nan_to_num(bicubic_interpolation(img, ori_pts))
        elif mode == 'nearest':
            value = np.nan_to_num(nearest_neighbor(img, ori_pts))
        canva[pts[1], pts[0],:] = value
    canva = np.clip(canva, 0, 255)
    ret = canva.astype(np.uint8)
    cv2.imwrite(f'rotate_{mode}.jpg', ret)


if __name__ == '__main__':

    embedded_img = cv2.imread('./image.jpg').astype(np.float32)
    img = cv2.imread('./board.jpg').astype(np.float32)
    emb_H = embedded_img.shape[0]
    emb_W = embedded_img.shape[1]
    pts1 = [(0, 0), (emb_W, 0), (emb_W, emb_H), (0, emb_H)]
    pts2 = [(253, 241), (413, 215), (413, 387), (253, 375)]
    start_time = time.time()
    rotate_img(img, angle=30) # clockwise 30 degree (in opencv coordinate, clockwise is positive)
    end_time = time.time() 
    print('rotate_img bilinear time: ', end_time - start_time)
    start_time = time.time()  
    rotate_img(img, angle=30, mode='bicubic')
    end_time = time.time()
    print('rotate_img bicubic time: ', end_time - start_time)
    start_time = time.time()
    rotate_img(img, angle=30, mode='nearest')
    end_time = time.time()
    print('rotate_img nearest time: ', end_time - start_time)
    start_time = time.time()
    warp_img(embedded_img, img, pts1, pts2)
    end_time = time.time()
    print('warp_img bilinear time: ', end_time - start_time)
    start_time = time.time()
    warp_img(embedded_img, img, pts1, pts2, mode='bicubic')
    end_time = time.time()
    print('warp_img bicubic time: ', end_time - start_time)
    start_time = time.time()
    warp_img(embedded_img, img, pts1, pts2, mode='nearest')
    end_time = time.time()
    print('warp_img nearest time: ', end_time - start_time)

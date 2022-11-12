import numpy as np

def filter_image(image):
    #return np.reshape(filter_image_2d(np.reshape(image, (50, 25))), (50*25))
    copy_image = image.copy()
    median = 0.5 * (np.max(image[20*25 : 30*25]) + np.min(image[20*25 : 30*25]))
    for i in range(image.shape[0]):
            if image[i] > median:
                copy_image[i] = 1
            else:
                copy_image[i] = 0
            
    return copy_image


def filter_image_2d(image):
    copy_image = image.copy()
    mid_image = image[20: 30, 10 : 20]
    median = 0.5 * (np.max(mid_image) + np.min(mid_image))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] > median:
                copy_image[i][j] = 1
            else:
                copy_image[i][j] = 0
            
    return copy_image

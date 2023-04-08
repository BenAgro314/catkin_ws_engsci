import cv2
import numpy as np

def blur_image_direction(image, direction, px):
    # Define the kernel size and shape based on the direction
    # Perform the convolution using the kernel
    if direction == 'POS_X':
        kernel = np.ones((1, px), np.uint8)
        blurred_image = cv2.dilate(image, kernel, iterations = 1, anchor = (px-1,0))
    elif direction == 'POS_Y':
        kernel = np.ones((px, 1), np.uint8)
        blurred_image = cv2.dilate(image, kernel, iterations = 1, anchor = (0, px-1))
    elif direction == 'NEG_X':
        kernel = np.ones((1, px), np.uint8)
        blurred_image = cv2.dilate(image, kernel, iterations =1, anchor =(0,0))
    elif direction == 'NEG_Y':
        kernel = np.ones((px, 1), np.uint8)
        blurred_image = cv2.dilate(image, kernel, iterations = 1, anchor =(0,0))
    else:
        raise ValueError("Invalid direction. Must be POS_X, POS_Y, NEG_X, or NEG_Y.")

    return blurred_image


if __name__ == "__main__":

    img = np.zeros((1000, 1000))
    img[400:500, 400:500] = 1

    cv2.imshow('before', img)
    

    blur_img = blur_image_direction(img, 'POS_Y', 101)
    cv2.imshow('after', blur_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
import cv2
import numpy as np

def process_image_with_squares(image_path, s1, s2, scale_factor):
    # Read the input image in grayscale
    image = cv2.imread(image_path, 1)
    gray = cv2.imread(image_path, 0)

    # Apply thresholding to the grayscale image
    th, threshed = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Filter the contours based on their area
    xcnts = []
    for contour in contours:
        if s1 < cv2.contourArea(contour) < s2:
            xcnts.append(contour)

    # Create a copy of the grayscale image to draw squares and lines on
    squares_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Draw a square around each dot and fill it with red color
    for contour in xcnts:
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate new dimensions by scaling the width and height
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)

        # Calculate new top-left coordinates to keep the square centered
        new_x = x - int((new_w - w) / 2)
        new_y = y - int((new_h - h) / 2)

        # Draw the enlarged square with red color
        cv2.rectangle(squares_image, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 0, 255), cv2.FILLED)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(squares_image, cv2.COLOR_BGR2HSV)

    # Define lower and upper thresholds for red color in HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Create a mask for the red color range
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Apply inverse binary thresholding to the mask
    _, mask_inv = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY_INV)

    # Create a copy of the original image to modify
    modified_image = squares_image.copy()

    # Set red pixels to white and non-red pixels to black
    modified_image[np.where(mask_inv == 0)] = (0, 0, 0)  # Set non-red pixels to black
    modified_image[np.where(mask_inv != 0)] = (255, 255, 255)  # Set red pixels to white

    # Find the coordinates where the black pixels start
    non_black_pixels = np.where(modified_image != [255, 255, 255])
    min_x = np.min(non_black_pixels[1])
    max_x = np.max(non_black_pixels[1])
    min_y = np.min(non_black_pixels[0])
    max_y = np.max(non_black_pixels[0])

    # Crop the modified image based on the coordinates where the black pixels start
    cropped_image = modified_image[min_y:max_y, min_x:max_x]

    # Convert the image to grayscale
    cropped_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Apply Prewitt edge detection
    edges_x = cv2.Sobel(cropped_gray, cv2.CV_64F, 1, 0, ksize=3)
    edges_y = cv2.Sobel(cropped_gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.addWeighted(edges_x, 0.5, edges_y, 0.5, 0)

    # Convert the edges to absolute values and rescale to 8-bit
    edges = cv2.convertScaleAbs(edges)

    # Find the coordinates of the first non-zero pixel in each row
    y_indices = np.argmax(edges > 0, axis=1)
    y_min = np.min(y_indices)
    y_max = np.max(y_indices)

    # Find the coordinates of the first non-zero pixel in each column
    x_indices = np.argmax(edges > 0, axis=0)
    x_min = np.min(x_indices)
    x_max = np.max(x_indices)

    # Crop the image based on the first edges encountered in each axis
    cropped_image = cropped_image[y_min:y_max+25, x_min:x_max+25]

    # Convert the cropped image to grayscale
    cropped_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Apply Harris corner detection
    corner_dst = cv2.cornerHarris(cropped_gray, 2, 3, 0.04)

    # Threshold the corner response to obtain the corners
    threshold = 0.05 * corner_dst.max()
    corners = np.argwhere(corner_dst > threshold)


    left_upper_corner = None
    right_upper_corner = None
    left_lower_corner = None
    right_lower_corner = None

    myMap = {}
    for corner in corners:
        y, x = corner[0], corner[1]
        cv2.circle(cropped_image, (x, y), 3, (0, 0, 255), -1)

        # Update the left-upper corner
        if left_upper_corner is None or (x <= left_upper_corner[1] and y <= left_upper_corner[0]):
            if(x in myMap):
                if(myMap[x] == y):
                    continue  
            myMap[x] = y
            left_upper_corner = (x, y)
        
        # Update the right-upper corner2
        if right_upper_corner is None or (x >= right_upper_corner[1] and y <= right_upper_corner[0]):
            if(x in myMap):
                if(myMap[x] == y):
                    continue
            right_upper_corner = (x, y)
            myMap[x] = y3
        # Update the left-lower corner
        if left_lower_corner is None or (x <= left_lower_corner[1] and y >= left_lower_corner[0]):
            if(x in myMap):
                if(myMap[x] == y):
                    continue
            left_lower_corner = (x, y)
            myMap[x] = y
        
        # Update the right-lower corner
        if right_lower_corner is None or (x >= right_lower_corner[1] and y >= right_lower_corner[0]):
            if(x in myMap):
                if(myMap[x] == y):
                    continue
            right_lower_corner = (x, y)
            myMap[x] = y

    # Print the coordinates of the corners
    print("Left-upper corner:", left_upper_corner)
    print("Right-upper corner:", right_upper_corner)
    print("Left-lower corner:", left_lower_corner)
    print("Right-lower corner:", right_lower_corner)


    x_coordinates = [left_upper_corner[0], right_upper_corner[0], left_lower_corner[0], right_lower_corner[0]]
    y_coordinates = [left_upper_corner[1], right_upper_corner[1], left_lower_corner[1], right_lower_corner[1]]

    min_x = min(x_coordinates)
    max_x = max(x_coordinates)
    min_y = min(y_coordinates)
    max_y = max(y_coordinates)

    for x in range(min_x, max_x - 10):
        for y in range(min_y, max_y - 10):
            pixel = squares_image[x, y]  # Access the pixel value at coordinates (x, y)
            if not np.array_equal(pixel, [0, 0, 255]):
                # If the pixel is not red (B < R and B < G), mark it as green
                image[x, y] = [0, 255, 0]  # Set pixel color to green (BGR 



    # Display the image with filled squares, red rectangle, and black regions
    cv2.imshow('Cropped Image with Corners', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Provide the image path, range for contour area filtering, and scale factor
image_path = 'ledbeyaz.jpg'
s1 = 1
s2 = 500
scale_factor = 2.6  # Adjust the scale factor as desired

# Process the image with squares and detect black regions within the red area
process_image_with_squares(image_path, s1, s2, scale_factor)

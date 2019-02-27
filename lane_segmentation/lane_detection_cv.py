import argparse
import numpy as np
import cv2

def fit_lanes(img, lines):
    # See https://github.com/davidawad/Lane-Detection
    
    # reshape lines to a 2d matrix
    lines = lines.reshape(lines.shape[0], lines.shape[2])
    # create array of slopes
    slopes = (lines[:,3] - lines[:,1]) /(lines[:,2] - lines[:,0])
    # remove junk from lists
    lines = lines[~np.isnan(lines) & ~np.isinf(lines)]
    slopes = slopes[~np.isnan(slopes) & ~np.isinf(slopes)]
    # convert lines into list of points
    lines.shape = (lines.shape[0]//2,2)

    # Right lane
    # move all points with negative slopes into right "lane"
    right_slopes = slopes[slopes < 0]
    right_lines = np.array(list(filter(lambda x: x[0] > (img.shape[1]/2), lines)))
    max_right_x, max_right_y = right_lines.max(axis=0)
    min_right_x, min_right_y = right_lines.min(axis=0)

    # Left lane
    # all positive  slopes go into left "lane"
    left_slopes = slopes[slopes > 0]
    left_lines = np.array(list(filter(lambda x: x[0] < (img.shape[1]/2), lines)))
    max_left_x, max_left_y = left_lines.max(axis=0)
    min_left_x, min_left_y = left_lines.min(axis=0)

    # Curve fitting approach
    # calculate polynomial fit for the points in right lane
    right_curve = np.poly1d(np.polyfit(right_lines[:,1], right_lines[:,0], 2))
    left_curve  = np.poly1d(np.polyfit(left_lines[:,1], left_lines[:,0], 2))

    # shared ceiling on the horizon for both lines
    min_y = min(min_left_y, min_right_y)

    # use new curve function f(y) to calculate x values
    max_right_x = int(right_curve(img.shape[0]))
    min_right_x = int(right_curve(min_right_y))

    min_left_x = int(left_curve(img.shape[0]))

    r1 = (min_right_x, min_y)
    r2 = (max_right_x, img.shape[0])
    right_lane = [r1, r2]

    l1 = (max_left_x, min_y)
    l2 = (min_left_x, img.shape[0])
    left_lane = [l1, l2]
    return [right_lane, left_lane]

def detect_lanes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    canny = cv2.Canny(np.uint8(blur), low_threshold, high_threshold)

    # Mask image
    h,w = gray.shape[:2]
    vertices = np.array([[(0, h), (int(0.4*w), int(0.6*h)), (int(0.6*w), int(0.6*h)), (w, h)]])
    mask = np.zeros((h,w))
    cv2.fillPoly(mask, vertices, 255)
    canny[mask == 0] = 0

    # Define the Hough transform parameters
    rho             = 2           # distance resolution in pixels of the Hough grid
    theta           = np.pi/180   # angular resolution in radians of the Hough grid
    threshold       = 15       # minimum number of votes (intersections in Hough grid cell)
    min_line_len    = 20       # minimum number of pixels making up a line
    max_line_gap    = 20       # maximum gap in pixels between connectable line segments
    lines = cv2.HoughLinesP(canny, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    # Fit lanes through lines
    lanes = fit_lanes(img, lines)

    # Draw lines
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)
    for line in lanes:
        cv2.line(img, line[0], line[1], (0,0,255), 8)

    return lanes, img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="filepath for image to mark", default='./test_images/solidWhiteRight.jpg')
    args = parser.parse_args()

    img = cv2.imread(args.file)
    lanes, lanes_debug = detect_lanes(img)

    cv2.imshow("", lanes_debug)
    cv2.waitKey(0)


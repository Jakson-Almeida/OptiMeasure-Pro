import cv2
import numpy as np

def find_bounding_polygon(contours):
    if not contours:
        return None

    # Find the convex hull of the contours
    hull = cv2.convexHull(np.vstack(contours))

    # Approximate the convex hull to get the quadrilateral
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    # Return the coordinates of the vertices as a list
    return approx.tolist() if len(approx) > 1 else [approx[0].tolist()]

# Callback function for the HSV sliders
def update_hsv_values(x):
    pass

# Try to initialize video capture from the webcam
cap = cv2.VideoCapture(0)

# If the webcam is not available, switch to video file
if not cap.isOpened():
    print("Webcam not found. Using video file.")
    cap.release()
    cap = cv2.VideoCapture('video.mp4')

# Create a window to display the HSV sliders
cv2.namedWindow('HSV Controls')

# Set initial values for the HSV sliders
h_min = 0
s_min = 208
v_min = 0
h_max = 255
s_max = 255
v_max = 255

# Create sliders for HSV values
cv2.createTrackbar('H Min', 'HSV Controls', h_min, 255, update_hsv_values)
cv2.createTrackbar('S Min', 'HSV Controls', s_min, 255, update_hsv_values)
cv2.createTrackbar('V Min', 'HSV Controls', v_min, 255, update_hsv_values)
cv2.createTrackbar('H Max', 'HSV Controls', h_max, 255, update_hsv_values)
cv2.createTrackbar('S Max', 'HSV Controls', s_max, 255, update_hsv_values)
cv2.createTrackbar('V Max', 'HSV Controls', v_max, 255, update_hsv_values)

# Initialize plot flags
plot_polygon = True
plot_contours = False
plot_both = True

while True:
    # Capture a frame from the video
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop the video when it ends
        continue

    # Get the values from the HSV sliders
    h_min = cv2.getTrackbarPos('H Min', 'HSV Controls')
    s_min = cv2.getTrackbarPos('S Min', 'HSV Controls')
    v_min = cv2.getTrackbarPos('V Min', 'HSV Controls')
    h_max = cv2.getTrackbarPos('H Max', 'HSV Controls')
    s_max = cv2.getTrackbarPos('S Max', 'HSV Controls')
    v_max = cv2.getTrackbarPos('V Max', 'HSV Controls')

    # Convert the frame to the HSV color space
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])

    # Create a mask using the defined threshold values
    mask = cv2.inRange(frame_hsv, lower_bound, upper_bound)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the detected lines on the original frame in green
    frame_lines = frame.copy()

    # Plot polygon and contours if enabled
    if plot_polygon or plot_contours:
        bounding_polygon = find_bounding_polygon(contours)
        if bounding_polygon is not None and len(bounding_polygon) >= 2:
            # Draw the bounding polygon using cv2.line
            for i in range(len(bounding_polygon) - 1):
                pt1 = tuple(bounding_polygon[i][0])
                pt2 = tuple(bounding_polygon[i + 1][0])
                cv2.line(frame_lines, pt1, pt2, (0, 0, 128), 2)
            # Draw the last line connecting the last and first points
            pt1 = tuple(bounding_polygon[-1][0])
            pt2 = tuple(bounding_polygon[0][0])
            cv2.line(frame_lines, pt1, pt2, (0, 0, 128), 2)

        # Plot contours if enabled
        if plot_contours:
            cv2.drawContours(frame_lines, contours, -1, (255, 0, 0), 2)

    # Display the frame with green lines, bounding polygon, and contours
    cv2.imshow('Video with Lines, Bounding Polygon, and Contours', frame_lines)

    # Display the mask
    cv2.imshow('Mask', mask)

    # Display the bounding polygon on the mask if both options are enabled
    # if plot_both:
    #     frame_mask = mask.copy()
    #     if bounding_polygon is not None and len(bounding_polygon) >= 2:
    #         # Draw the bounding polygon using cv2.line
    #         for i in range(len(bounding_polygon) - 1):
    #             pt1 = tuple(bounding_polygon[i][0])
    #             pt2 = tuple(bounding_polygon[i + 1][0])
    #             cv2.line(frame_mask, pt1, pt2, (0, 0, 128), 2)
    #         # Draw the last line connecting the last and first points
    #         pt1 = tuple(bounding_polygon[-1][0])
    #         pt2 = tuple(bounding_polygon[0][0])
    #         cv2.line(frame_mask, pt1, pt2, (0, 0, 128), 2)
    #     cv2.imshow('Mask with Bounding Polygon', frame_mask)

    # Press the 'q' key to exit the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        plot_polygon = not plot_polygon
    elif key == ord('c'):
        plot_contours = not plot_contours
    elif key == ord('1'):
        plot_polygon = True
        plot_contours = True
        plot_both = True
    elif key == ord('2'):
        plot_polygon = True
        plot_contours = False
        plot_both = True

# Release the video capture, close the windows, and exit the program
cap.release()
cv2.destroyAllWindows()

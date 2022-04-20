import cv2  # Import the OpenCV library to enable computer vision
import numpy as np  # Import the NumPy scientific computing library
import edge_detection as edge  # Handles the detection of lane lines
import matplotlib.pyplot as plt


# Make sure the video file is in the same directory as your code
#filename = 'project_video.mp4'
filename = 'challenge_video.mp4'
#filename = 'harder_challenge_video.mp4'


file_size = (1920, 1080)  # Assumes 1920x1080 mp4
scale_ratio = 1  # Option to scale to fraction of original size.

# We want to save the output to a video file
output_filename = 'orig_lane_detection_1_lanes.mp4'
output_frames_per_second = 20.0

# Global variables
prev_leftx = None
prev_lefty = None
prev_rightx = None
prev_righty = None
prev_left_fit = []
prev_right_fit = []

prev_leftx2 = None
prev_lefty2 = None
prev_rightx2 = None
prev_righty2 = None
prev_left_fit2 = []
prev_right_fit2 = []
centerOfLane = 500 # --- initialValue ---
carPosition = 640 # --- initially ---



class Lane:
    """
    Represents a lane on a road.
    """

    def __init__(self, orig_frame):

        self.orig_frame = orig_frame

        # This will hold an image with the lane lines
        self.lane_line_markings = None

        # This will hold the image after perspective transformation
        self.warped_frame = None
        self.transformation_matrix = None
        self.inv_transformation_matrix = None

        # (Width, Height) of the original video frame (or image)
        self.orig_image_size = self.orig_frame.shape[::-1][1:]

        width = self.orig_image_size[0]
        height = self.orig_image_size[1]
        self.width = width
        self.height = height

    def perspective_transform(self, frame=None, plot=False):

        if frame is None:
            frame = self.lane_line_markings

        # Calculate the transformation matrix
        self.transformation_matrix = cv2.getPerspectiveTransform(
            self.roi_points, self.desired_roi_points)

        # Calculate the inverse transformation matrix
        self.inv_transformation_matrix = cv2.getPerspectiveTransform(
            self.desired_roi_points, self.roi_points)

        # Perform the transform using the transformation matrix
        self.warped_frame = cv2.warpPerspective(
            frame, self.transformation_matrix, self.orig_image_size, flags=(
                cv2.INTER_LINEAR))

        # Convert image to binary
        (thresh, binary_warped) = cv2.threshold(
            self.warped_frame, 172, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
        self.warped_frame = binary_warped

        # Display the perspective transformed (i.e. warped) frame
        if plot == True:
            warped_copy = self.warped_frame.copy()
            warped_plot = cv2.polylines(warped_copy, np.int32([
                self.desired_roi_points]), True, (147, 20, 255), 3)

            # Display the image
            while (1):
                cv2.imshow('Warped Image', warped_plot)

                # Press any key to stop
                if cv2.waitKey(0):
                    break

            cv2.destroyAllWindows()

        return self.warped_frame

    def calculate_car_position(self, print_to_terminal=False):

        car_location = self.orig_frame.shape[1] / 2

        # Fine the x coordinate of the lane line bottom
        height = self.orig_frame.shape[0]
        bottom_left = self.left_fit[0] * height ** 2 + self.left_fit[
            1] * height + self.left_fit[2]
        bottom_right = self.right_fit[0] * height ** 2 + self.right_fit[
            1] * height + self.right_fit[2]

        center_lane = (bottom_right - bottom_left) / 2 + bottom_left
        center_offset = (np.abs(car_location) - np.abs(
            center_lane)) * self.XM_PER_PIX * 100
        # ---
        global centerOfLane
        global carPosition
        carPosition = car_location
        if center_lane < 600 or center_lane >450:
            centerOfLane = center_lane
        else:
            centerOfLane = 500

        if print_to_terminal == True:

            print("Car",str(center_offset) + 'cm')
            print("Lane",str(center_lane) + 'cm')
            print("CarLocation",str(car_location) + 'cm')

        self.center_offset = center_offset

        return center_offset

    def calculate_curvature(self, print_to_terminal=False):

        y_eval = np.max(self.ploty)

        # Fit polynomial curves to the real world environment
        left_fit_cr = np.polyfit(self.lefty * self.YM_PER_PIX, self.leftx * (
            self.XM_PER_PIX), 2)
        right_fit_cr = np.polyfit(self.righty * self.YM_PER_PIX, self.rightx * (
            self.XM_PER_PIX), 2)

        # Calculate the radii of curvature
        left_curvem = ((1 + (2 * left_fit_cr[0] * y_eval * self.YM_PER_PIX + left_fit_cr[
            1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        right_curvem = ((1 + (2 * right_fit_cr[
            0] * y_eval * self.YM_PER_PIX + right_fit_cr[
                                  1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

        # Display on terminal window
        if print_to_terminal == True:
            print(left_curvem, 'm', right_curvem, 'm')

        self.left_curvem = left_curvem
        self.right_curvem = right_curvem

        return left_curvem, right_curvem

    def display_curvature_offset(self, frame=None, plot=False):

        image_copy = None
        if frame is None:
            image_copy = self.orig_frame.copy()
        else:
            image_copy = frame

        cv2.putText(image_copy, 'Curve Radius: ' + str((
                                                               self.left_curvem + self.right_curvem) / 2)[:7] + ' m',
                    (int((
                                 5 / 600) * self.width), int((
                                                                     20 / 338) * self.height)),
                    cv2.FONT_HERSHEY_SIMPLEX, (float((
                                                             0.5 / 600) * self.width)), (
                        255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image_copy, 'Center Offset: ' + str(
            self.center_offset)[:7] + ' cm', (int((
                                                          5 / 600) * self.width), int((
                                                                                              40 / 338) * self.height)),
                    cv2.FONT_HERSHEY_SIMPLEX, (float((
                                                             0.5 / 600) * self.width)), (
                        255, 255, 255), 2, cv2.LINE_AA)

        if plot == True:
            cv2.imshow("Image with Curvature and Offset", image_copy)

        return image_copy

    def calculate_histogram(self, frame=None, plot=True):
        if frame is None:
            frame = self.warped_frame

        # Generate the histogram
        self.histogram = np.sum(frame[int(
            frame.shape[0] / 2):, :], axis=0)

        if plot == True:
            # Draw both the image and the histogram
            figure, (ax1, ax2) = plt.subplots(2, 1)  # 2 row, 1 columns
            figure.set_size_inches(10, 5)
            ax1.imshow(frame, cmap='gray')
            ax1.set_title("Warped Binary Frame")
            ax2.plot(self.histogram)
            ax2.set_title("Histogram Peaks")
            plt.show()

        return self.histogram

# --- main function to run ---
def main():
    # Load a video
    cap = cv2.VideoCapture(filename)

    # --- save the output video ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    result = cv2.VideoWriter(output_filename,
                             fourcc,
                             output_frames_per_second,
                             file_size)

    # Process the video
    while cap.isOpened():

        # Capture one frame at a time
        success, frame = cap.read()

        if success:

            # Resize the frame
            width = int(frame.shape[1] * scale_ratio)
            height = int(frame.shape[0] * scale_ratio)
            frame = cv2.resize(frame, (width, height))

            # Store the original frame
            original_frame = frame.copy()

            # Create a Lane object
            lane_obj = Lane(orig_frame=original_frame)

            warped_frame = lane_obj.perspective_transform(plot=False)


            # ---
            # Generate the image histogram to serve as a starting point
            # for finding lane line pixels
            histogram = lane_obj.calculate_histogram(plot=False)
            # Calculate lane line curvature (left and right lane lines)
            lane_obj.calculate_curvature(print_to_terminal=False)

            # Calculate center offset
            lane_obj.calculate_car_position(print_to_terminal=False)

            # Display curvature and center offset on image
            frame_with_lane_lines2 = lane_obj.display_curvature_offset(
                frame=frame_with_lane_lines, plot=False)
            # ---

            # Write the frame to the output video file
            result.write(frame_with_lane_lines2)

            # Display the frame
            cv2.imshow("Frame", frame_with_lane_lines2)

            # Display frame for X milliseconds and check if q key is pressed
            # q == quit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # No more video frames left
        else:
            break

    # Stop when the video is finished
    cap.release()

    # Release the video recording
    result.release()

    # Close all windows
    cv2.destroyAllWindows()


main()

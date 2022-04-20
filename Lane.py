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
        """
          Default constructor

        :param orig_frame: Original camera image (i.e. frame)
        """
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
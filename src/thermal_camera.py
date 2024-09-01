
#!/usr/bin/env python3

from dataclasses import dataclass
import io
import cv2
import numpy as np

@dataclass
class Point:
    """
    A data class representing a point in 2D space with an associated temperature.

    Attributes:
        x_pos (int): The x-coordinate of the point.
        y_pos (int): The y-coordinate of the point.
        temperature (float): The temperature value associated with this point.
    """
    x_pos: int
    y_pos: int
    temperature: float

class DynamicList(list):
    """ A data structure to act as a list, with the added functionality of removing 
    one of the list elements."""

    def remove(self, index):
        """
        Removes the element at the specified index from this list.

        Args:
            index (int): The index of the element to be removed. Index must be within range (0 <= index < len(self)).
        Raises:
            IndexError: If the index is out of range (i.e., not in the range [0, len(self)-1]).
        Returns:
            None
        """     

        if 0 <= index < len(self):
            super().pop(index)
        else:
            raise IndexError("Index out of range")

class ThermalCamera:
    """Class to represent the thermal camera object

    Raises:
        IOError: If a camera device could not be opened.
        IOError: If an empty frame was received.

    """

    COLORMAPS = [
        {'name' : 'Jet', "cv_map"  : cv2.COLORMAP_JET},
        {'name' : 'Hot', "cv_map" : cv2.COLORMAP_HOT},
        {'name' : 'Magma', "cv_map" : cv2.COLORMAP_MAGMA},
        {'name' : 'Inferno', "cv_map" : cv2.COLORMAP_INFERNO},
        {'name' : 'Plasma', "cv_map" : cv2.COLORMAP_PLASMA},
        {'name' : 'Bone', "cv_map" : cv2.COLORMAP_BONE},
        {'name' : 'Spring', "cv_map" : cv2.COLORMAP_SPRING},
        {'name' : 'Autumn', "cv_map" : cv2.COLORMAP_AUTUMN},
        {'name' : 'Viridis', "cv_map" : cv2.COLORMAP_VIRIDIS},
        {'name' : 'Parula', "cv_map" : cv2.COLORMAP_PARULA},
        {'name' : 'InvRainbow', "cv_map" : cv2.COLORMAP_RAINBOW}
    ]

    def __init__(self, device, scale, alpha, colormap_name):
        """Initializes the thermal camera

        Args:
            device (str): Path to the video device eg. /dev/videoX on Linux
            scale (float): The scale with which to scale the camera image
            alpha (float): The contrast correction value
            colormap_name (str): Name of the colormap to apply
        """

        # We need to know if we are running on the Pi,
        # because openCV behaves a little oddly on all the builds!
        # https://raspberrypi.stackexchange.com/questions/5100/detect-that-a-python-program-is-running-on-the-pi
        self.is_pi = self._is_raspberry_pi()

        # Initialize the video stream
        self.cap = cv2.VideoCapture(f'/dev/video{device}', cv2.CAP_V4L)

        # Pull in the video but do NOT automatically convert to RGB,
        # else it breaks the temperature data!
        # https://stackoverflow.com/questions/63108721/opencv-setting-videocap-property-to-cap-prop-convert-rgb-generates-weird-boolean
        if self.is_pi:
            self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
        else:
            # For some systems 0.0 need to be replaced by a boolean
            self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)

        # Define the settings
        self.sensor_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #Sensor width

        # Sensor height - only half, the other half is used for thermal data
        self.sensor_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)
        self.scale = scale
        self.scaled_width = self.sensor_width * self.scale
        self.scaled_height = self.sensor_height * self.scale
        self.alpha = alpha
        self.colormap_name = colormap_name

        # Extract the list index of the given colormap name
        self.colormap_index = self._find_colormap_index(self.colormap_name)
        self.blur_radius = 0
        self.threshold = 2

        # Define the attributes
        self.center_point = Point(int(self.sensor_width/2), int(self.sensor_height/2), 0)
        self.min_point = Point(0, 0, 0)
        self.max_point = Point(0, 0, 0)

        # Initialize a dynamic list to store user points
        self.user_points = DynamicList()

        # Initialize a variable to store the average temperature
        self.avg_temp = 0

    def capture_status(self):
        """Get the status of the capture object

        Returns:
            boolean: True if the capture is open, false if not.
        """
        return self.cap.isOpened()

    def stop_capture(self):
        """Stops the capture and releases the camera.
        """
        self.cap.release()

    def get_frame(self):
        """Retrieves a frame from the camera and converts it to temperatures, 
        applies scaling and color mapping. Extracts min, max, average, center temperatures
        and temperatures for the user specified points.

        Raises:
            IOError: If a camera device could not be opened.
            IOError: If an empty frame was received.

        Returns:
            UMat: The colormapped frame
        """
        if not self.cap.isOpened():
            raise IOError("The capture device is not open!")

        # Capture frame
        ret, frame = self.cap.read()
        if not ret:
            raise IOError("Received empty frame!")

        # Split the frame into image data and thermal data
        imdata, thdata = np.array_split(frame, 2) # pylint: disable=W0632

        # Now parse the data from the bottom frame and convert to temp!
        # https://www.eevblog.com/forum/thermal-imaging/infiray-and-their-p2-pro-discussion/200/
        # Huge props to LeoDJ for figuring out how the data is 
        # stored and how to compute temp from it.
        # grab data from the center pixel...

        # Calculate temperatures for the entire frame
        thermal_matrix = thdata[...,1].astype('uint16') * 256
        thermal_matrix = thermal_matrix + thdata[..., 0]
        thermal_matrix = thermal_matrix/64 - 273.15
        thermal_matrix = thermal_matrix.round(2)

        # Extract different temperature points
        self.center_point = self._extract_center_temp(thermal_matrix)
        self.max_point = self._extract_max_temp(thermal_matrix)
        self.min_point = self._extract_min_temp(thermal_matrix)
        self.avg_temp = self._extract_avg_temp(thermal_matrix)

        # Extract user points
        for point in self.user_points:
            point.temperature = thermal_matrix[point.y_pos][point.x_pos]

        # Convert real image to RGB
        bgr = cv2.cvtColor(imdata,  cv2.COLOR_YUV2BGR_YUYV)
        # Set the contrast
        bgr = cv2.convertScaleAbs(bgr, alpha=self.alpha)#Contrast
        # Bicubic interpolate, upscale and blur
        bgr = cv2.resize(bgr,(self.scaled_width, self.scaled_height),
                            interpolation=cv2.INTER_CUBIC)#Scale up!
        if self.blur_radius > 0:
            bgr = cv2.blur(bgr, (self.blur_radius, self.blur_radius))

        # Apply the colormap
        heatmap = self._apply_colormap(bgr)

        return heatmap

    def increase_scaling(self):
        """Increase the scaling of the image with 1. Max. 5.
        """
        self.scale += 1
        self.scale = min(self.scale, 5)

        self.scaled_width = self.sensor_width * self.scale
        self.scaled_height = self.sensor_height * self.scale

    def decrease_scaling(self):
        """Decrease the scaling of an image with 1. Min. 1.
        """
        self.scale -= 1
        self.scale = max(self.scale, 1)
        self.scaled_width = self.sensor_width * self.scale
        self.scaled_height = self.sensor_height * self.scale

    def increase_blur(self):
        """Increase the blur of the frame.
        """
        self.blur_radius += 1

    def decrease_blur(self):
        """Decrease the blur of the frame.
        """
        self.blur_radius -= 1
        self.blur_radius = max(self.blur_radius, 0)

    def increase_threshold(self):
        """Increase the threshold value.
        """
        self.threshold += 1

    def decrease_threshold(self):
        """Decrease the threshold value.
        """
        self.threshold -= 1
        self.threshold = max(self.threshold, 0)

    def increase_contrast(self):
        """Increase the contrast value.
        """
        self.alpha += 0.1
        self.alpha = round(self.alpha, 1)#fix round error
        self.alpha = min(self.alpha, 3.0)

    def decrease_contrast(self):
        """Decrease the threshold value.
        """
        self.alpha -= 0.1
        self.alpha = round(self.alpha, 1)#fix round error
        self.alpha = max(self.alpha, 0.0)

    def next_colormap(self):
        """Apply the next colormap from the list.
        """
        self.colormap_index += 1
        if self.colormap_index == 11:
            self.colormap_index = 0

        return self.COLORMAPS[self.colormap_index]["name"]

    def add_point(self, x: int, y: int):
        """Add a point for user temperature monitoring

        Args:
            x (int): x position within the frame (camera sensor coordinates)
            y (int): y position within the frame (camera sensor coordinates)
        """

        self.user_points.append(Point(x, y, 0))

    def remove_point(self, point_number:int):
        """Removes a point from the user temperature monitoring list.

        Args:
            point_number (int): The index of the point to remove
        """
        if point_number >= 0 and point_number < len(self.user_points):
            self.user_points.remove(point_number)

    def _find_colormap_index(self, colormap_name):
        """Finds the index of the colormap in the COLORMAPS list,
        given the name of the colormap

        Args:
            colormap_name (str): Name of the colormap

        Returns:
            int: Index of the colormap or None if the specified name was not found.
        """
        for colormap_index, map_dict in enumerate(self.COLORMAPS):
            if map_dict['name'] == colormap_name:
                return colormap_index

        return None

    def _apply_colormap(self, image):
        """Applies the currently selected colormap to the frame

        Args:
            image (UMat): The frame

        Returns:
            UMat: The frame with the applied colormap
        """
        colormap = self.COLORMAPS[self.colormap_index]['cv_map']

        image = cv2.applyColorMap(image, colormap)
        if self.COLORMAPS[self.colormap_index]['name'] == "InvRaibow":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def _extract_center_temp(self, thdata):
        """Extract the temperature of the center of the image

        Args:
            thdata (ndarray): 2-dimensional array with thermal data

        Returns:
            float: Temperature of the image center
        """
        self.center_point.temperature = thdata[self.center_point.y_pos][self.center_point.x_pos]
        return  self.center_point

    def _extract_max_temp(self, thdata):
        """Extract the maximum temperature of the image.

        Args:
            thdata (ndarray): 2-dimensional array with thermal data

        Returns:
            Point: The Point object, that contains the coordinates and the temperature.
        """
        max_val = thdata.max()
        
        # This will return array, as multiple points 
        # might have the same value, we will just extract the first point
        mcol, mrow = np.where((thdata == max_val))
        mrow = mrow[0]
        mcol = mcol[0]

        return Point(mcol, mrow, max_val)

    def _extract_min_temp(self, thdata):
        """Extract the minimum temperature of the image.

        Args:
            thdata (ndarray): 2-dimensional array with thermal data

        Returns:
            Point: The Point object, that contains the coordinates and the temperature.
        """        
        min_val = thdata.min()
        
        # This will return array, as multiple points 
        # might have the same value, we will just extract the first point
        mcol, mrow = np.where((thdata == min_val))
        mrow = mrow[0]
        mcol = mcol[0]

        return Point(mcol, mrow, min_val)

    def _extract_avg_temp(self, thdata):
        """Extract the average temperature of the image.

        Args:
            thdata (ndarray): 2-dimensional array with thermal data

        Returns:
            float: The average temperature of the image.
        """        
        temp = round(thdata.mean(), 2)
        return temp

    def _is_raspberry_pi(self):
        """Determines if this software is running on a Raspberry Pi

        Returns:
            boolean: True if it is running on a Raspberry pi, false otherwise.
        """
        try:
            with io.open('/sys/firmware/devicetree/base/model', 'r', encoding='utf-8') as m:
                if 'raspberry pi' in m.read().lower():
                    return True
        except Exception:
            pass
        return False

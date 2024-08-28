
#!/usr/bin/env python3

from dataclasses import dataclass
import io
import cv2
import numpy as np

@dataclass
class Point:    
    x_pos: int
    y_pos: int
    temperature: float


class ThermalCamera:
    
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
        self.center_point = Point(0, int(self.sensor_width/2), int(self.sensor_height/2))
        self.min_point = Point(0, 0, 0)
        self.max_point = Point(0, 0, 0)

        self.user_points = []

        self.avg_temp = 0

    def capture_status(self):
        return self.cap.isOpened()

    def stop_capture(self):
        self.cap.release()

    def get_frame(self):
        if not self.cap.isOpened():
            raise IOError("The capture device is not open!")

        # Capture frame
        ret, frame = self.cap.read()
        if not ret:
            raise IOError("Received empty frame!")

        # Split the frame into image data and thermal data
        imdata, thdata = np.array_split(frame, 2) # pylint: disable=W0632

        #now parse the data from the bottom frame and convert to temp!
        #https://www.eevblog.com/forum/thermal-imaging/infiray-and-their-p2-pro-discussion/200/
        #Huge props to LeoDJ for figuring out how the data is stored and how to compute temp from it.
        #grab data from the center pixel...

        # Extract different temperatures
        self.center_point = self._extract_center_temp(thdata)
        self.max_point = self._extract_max_temp(thdata)
        self.min_point = self._extract_min_temp(thdata)
        self.avg_temp = self._extract_avg_temp(thdata)

        # Extract user points
        for point in self.user_points:
            point = self._extract_point_temperature(thdata, point)

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
        self.scale += 1
        self.scale = min(self.scale, 5)

        self.scaled_width = self.sensor_width * self.scale
        self.scaled_height = self.sensor_height * self.scale

    def decrease_scaling(self):
        self.scale -= 1
        self.scale = max(self.scale, 1)
        self.scaled_width = self.sensor_width * self.scale
        self.scaled_height = self.sensor_height * self.scale

    def increase_blur(self):
        self.blur_radius += 1

    def decrease_blur(self):
        self.blur_radius -= 1
        self.blur_radius = max(self.blur_radius, 0)

    def increase_threshold(self):
        self.threshold += 1

    def decrease_threshold(self):
        self.threshold -= 1
        self.threshold = max(self.threshold, 0)

    def increase_contrast(self):
        self.alpha += 0.1
        self.alpha = round(self.alpha, 1)#fix round error
        self.alpha = min(self.alpha, 3.0)

    def decrease_contrast(self):
        self.alpha -= 0.1
        self.alpha = round(self.alpha, 1)#fix round error
        self.alpha = max(self.alpha, 0.0)

    def next_colormap(self):

        self.colormap_index += 1
        if self.colormap_index == 11:
            self.colormap_index = 0
        
        return self.COLORMAPS[self.colormap_index]["name"]
            
    def _find_colormap_index(self, colormap_name):
        for colormap_index, map_dict in enumerate(self.COLORMAPS):
            if map_dict['name'] == colormap_name:
                return colormap_index
        
        return None

    def _apply_colormap(self, image):
        #apply colormap
        colormap = self.COLORMAPS[self.colormap_index]['cv_map']
        
        image = cv2.applyColorMap(image, colormap)
        if self.COLORMAPS[self.colormap_index]['name'] == "InvRaibow":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def _calculate_point_temperature(self, hi, lo):
        lo = lo * 256
        rawtemp = hi + lo
        temp = (rawtemp/64.0) - 273.15
        temp = round(temp, 2)
        return temp

    def _extract_point_temperature(self, thdata, point):
        hi = int(thdata[point.x_pos][point.y_pos][0])
        lo = int(thdata[point.x_pos][point.y_pos][1])

        point.temperature = self._calculate_point_temperature(hi, lo)

        return point

    def _extract_center_temp(self, thdata):
        return self._extract_point_temperature(thdata, self.center_point)

    def _extract_max_temp(self, thdata):
        #find the max temperature in the frame
        lomax = int(thdata[...,1].max())
        posmax = int(thdata[...,1].argmax())
        #since argmax returns a linear index, convert back to row and col
        mcol,mrow = divmod(posmax, self.sensor_width)
        himax = int(thdata[mcol][mrow][0])

        temp = self._calculate_point_temperature(himax, lomax)

        return Point(mcol, mrow, temp)

    def _extract_min_temp(self, thdata):
        #find the lowest temperature in the frame
        lomin = int(thdata[...,1].min())
        posmin = int(thdata[...,1].argmin())
        #since argmax returns a linear index, convert back to row and col
        lcol,lrow = divmod(posmin, self.sensor_width)
        himin = int(thdata[lcol][lrow][0])

        temp = self._calculate_point_temperature(himin, lomin)

        return Point(lcol, lrow, temp)

    def _extract_avg_temp(self, thdata):
        #find the average temperature in the frame
        loavg = int(thdata[...,1].mean())
        hiavg = int(thdata[...,0].mean())
        
        temp = self._calculate_point_temperature(hiavg, loavg)

        return temp

    def _is_raspberry_pi(self):
        try:
            with io.open('/sys/firmware/devicetree/base/model', 'r', encoding='utf-8') as m:
                if 'raspberry pi' in m.read().lower():
                    return True
        except Exception:
            pass
        return False

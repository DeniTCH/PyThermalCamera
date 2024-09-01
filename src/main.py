#!/usr/bin/env python3

'''
Les Wright 21 June 2023,
https://youtube.com/leslaboratory

Refactored by @DeniTCH August 2024

A Python program to read, parse and display thermal data from the
Topdon TC001 or IniRay P2Pro Thermal camera!
'''

import time
import cv2
import click
from thermal_camera import ThermalCamera

class ThermalApp:
    """Class to represent the thermal camera application and act as a container for
        all its functions.
    """
    WINDOW_NAME = 'Thermal'

    def __init__(self, thermal_camera):

        self.thermal_camera = thermal_camera
        self.is_pi = self.thermal_camera.is_pi

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fullscreen = False
        self.hud = True

        self.video_handle = None
        self.recording = False
        self.recording_start_time = None
        self.elapsed_time = "00:00:00"
        self.snaptime = "None"

        self.point_erase_mode = False

        # GUI texts
        self.gui_colormap_text = self.thermal_camera.colormap_name

        # Create the GUI window
        self._create_window()

        # Start capture
        self._start_capture()

    def _start_capture(self):
        """ Main program loop """

        while self.thermal_camera.capture_status():
            frame = self.thermal_camera.get_frame()

            # Draw the GUI elements and temperature markers
            self._draw_gui(frame)

            # Display the image in the window
            cv2.imshow(self.WINDOW_NAME, frame)

            # If we are recording
            if self.recording:
                self._handle_recording(frame)

            # Handle key input
            self._handle_keyboard_input(frame)

    @staticmethod
    def _handle_mouse_input(event, x, y, _, param):
        if event == cv2.EVENT_LBUTTONDOWN:

            # Convert the coordinates to image coordinates
            x_pos = int(x / param.thermal_camera.scale)
            y_pos = int(y / param.thermal_camera.scale)
            print(f'Added user monitor point at ({x_pos}, {y_pos})')
            param.thermal_camera.add_point(x_pos, y_pos)

    def _handle_keyboard_input(self, image):
        """ Handles key input

        Args:
            image (UMat): The frame data.
        """
        #TODO: Load keymap from a configuration file

        #FIXME: This solution does not seem to work in GNOME?
        # Detect if the window has been closed with the X of the window
        if cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1:
            #print(cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE))
            key = cv2.waitKey(1)

            if key == ord('o'):
                if self.point_erase_mode:
                    self.point_erase_mode = False
                    print('Exited point erase mode')
                else:
                    self.point_erase_mode = True
                    print('Entered point erase mode')


            if self.point_erase_mode and key >= 48 and key <= 57:
                number = int(chr(key))
                self.thermal_camera.remove_point(number)
                print(f'Removed user monitor point #{number}')
                self.point_erase_mode = False

            if key == ord('a'): #Increase blur radius
                self.thermal_camera.increase_blur()
            if key == ord('z'): #Decrease blur radius
                self.thermal_camera.decrease_blur()

            if key == ord('s'): #Increase threshold
                self.thermal_camera.increase_threshold()
            if key == ord('x'): #Decrease threashold
                self.thermal_camera.decrease_threshold()

            if key == ord('d'): #Increase scale
                self.thermal_camera.increase_scaling()
                if not self.fullscreen and not self.is_pi:
                    cv2.resizeWindow(self.WINDOW_NAME,
                                    self.thermal_camera.scaled_width,
                                    self.thermal_camera.scaled_height)

            if key == ord('c'): #Decrease self.thermal_camera.scale
                self.thermal_camera.decrease_scaling()
                if not self.fullscreen and not self.is_pi:
                    cv2.resizeWindow(self.WINDOW_NAME,
                                    self.thermal_camera.scaled_width,
                                    self.thermal_camera.scaled_height)

            if key == ord('e'): #enable fullscreen
                self.fullscreen = True
                cv2.namedWindow(self.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            if key == ord('w'): #disable fullscreen
                self.fullscreen = False
                cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)
                cv2.setWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_AUTOSIZE,cv2.WINDOW_GUI_NORMAL)
                cv2.resizeWindow(self.WINDOW_NAME,
                                self.thermal_camera.scaled_width,
                                self.thermal_camera.scaled_height)

            if key == ord('f'): # Increase contrast
                self.thermal_camera.increase_contrast()
            if key == ord('v'): # Decrease contrast
                self.thermal_camera.decrease_contrast()

            # Hud on/off
            if key == ord('h'):
                if self.hud:
                    self.hud=False
                else:
                    self.hud=True

            if key == ord('m'): #m to cycle through color maps
                self.gui_colormap_text = self.thermal_camera.next_colormap()

            if key == ord('r') and not self.recording: #r to start reording
                self.video_handle = self._start_recording()
                self.recording = True
                self.recording_start_time = time.time()
            if key == ord('t'): #f to finish reording
                self.recording = False
                self.elapsed_time = "00:00:00"

            if key == ord('p'): #f to finish reording
                self.snaptime = self._snapshot(image)

            if key == ord('q'):
                self.thermal_camera.stop_capture()
                self.thermal_camera.stop_capture()
                cv2.destroyAllWindows()
        else:
            self.thermal_camera.stop_capture()
            cv2.destroyAllWindows()

    def _handle_recording(self, image):
        """ Appends a frame to the video stream ad updates the elapsed time.

        Args:
            image (UMat): The frame data.
        """
        self.elapsed_time = time.time() - self.recording_start_time
        self.elapsed_time = time.strftime("%H:%M:%S", time.gmtime(self.elapsed_time))
        #print(elapsed)
        self.video_handle.write(image)

    def _draw_gui(self, image):
        """ Draws the GUI

        Args:
            image (UMat): The frame data.
        """
        # Draw center crosshairs
        self._draw_crosshairs(image, self.thermal_camera.center_point)

        # Draw user points
        for idx, point in enumerate(self.thermal_camera.user_points):
            self._draw_crosshairs(image, point, point_name=f'P{idx}', crosshair_size=10)

        # Show hud
        if self.hud:
            self._draw_hud(image)

        # Display floating max temperature
        threshold = self.thermal_camera.threshold
        if self.thermal_camera.max_point.temperature > self.thermal_camera.avg_temp + threshold:
            self._draw_circle(image, self.thermal_camera.max_point, (0, 0, 255))

        # Display floating min temperature
        if self.thermal_camera.min_point.temperature < self.thermal_camera.avg_temp - threshold:
            self._draw_circle(image,
                            self.thermal_camera.min_point,
                            (255, 0, 0))

    def _draw_circle(self, image, point, color):
        """ Draws a circle of the specified color at the sp

        Args:
            image (UMat): The frame data
            position (tuple): The circle location in image coordinates,
            color (tuple): Circle color in (B,G,R) format.
            temp (float): The temperature value to display next to the circle.
        """
        col = point.x_pos
        row = point.y_pos
        temp = point.temperature
        scale = self.thermal_camera.scale

        cv2.circle(image, (row*scale, col*scale), 5, (0,0,0), 2)
        cv2.circle(image, (row*scale, col*scale), 5, color, -1)
        cv2.putText(image,str(temp)+' C', ((row*scale)+10, (col*scale)+5),\
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,0,0), 2, cv2.LINE_AA)
        cv2.putText(image,str(temp)+' C', ((row*scale)+10, (col*scale)+5),\
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 255, 255), 1, cv2.LINE_AA)

    def _draw_hud(self, image):
        """ Draws the HUD box in the right corner of the
           provided frame

        Args:
            image (UMat): The frame data.
        """
        # Display black box for our data
        cv2.rectangle(image, (0, 0), (160, 120), (0,0,0), -1)
        # Put text in the box
        cv2.putText(image, f'Avg Temp: {self.thermal_camera.avg_temp} C', (10, 14),\
        self.font, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(image, f'Label Threshold: {self.thermal_camera.threshold} C', (10, 28),\
        self.font, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(image, f'Colormap: {self.gui_colormap_text}', (10, 42),\
        self.font, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(image, f'Blur: {self.thermal_camera.blur_radius}', (10, 56),\
        self.font, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(image, f'Scaling: {self.thermal_camera.scale}', (10, 70),\
        self.font, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(image, f'Contrast: {self.thermal_camera.alpha}', (10, 84),\
        self.font, 0.4,(0, 255, 255), 1, cv2.LINE_AA)


        cv2.putText(image,f'Snapshot: {self.snaptime}', (10, 98),\
        self.font, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

        if not self.recording:
            cv2.putText(image, f'Recording: {self.elapsed_time}', (10, 112),\
            self.font, 0.4,(200, 200, 200), 1, cv2.LINE_AA)
        else:
            cv2.putText(image, f'Recording: {self.elapsed_time}', (10, 112),\
            self.font, 0.4,(40, 40, 255), 1, cv2.LINE_AA)


    def _draw_crosshairs(self, image, point, point_name=None, crosshair_size=20):
        """ Draws crosshairs in the specified poitn and
        frame and show the temperature of the point

        Args:
            image (UMat): The frame data
            point (Point): The point object
        """
        scale = self.thermal_camera.scale

        # Draw the crosshairs
        cv2.line(image, (point.x_pos * scale, point.y_pos * scale + crosshair_size),\
        (point.x_pos * scale, point.y_pos * scale - crosshair_size), (255, 255, 255), 2) #vline
        cv2.line(image, (point.x_pos * scale + crosshair_size, point.y_pos * scale) ,\
        (point.x_pos * scale - crosshair_size, point.y_pos * scale) ,(255,255,255), 2) #hline

        cv2.line(image,(point.x_pos * scale, point.y_pos * scale + crosshair_size),\
        (point.x_pos * scale, point.y_pos * scale- crosshair_size), (0, 0, 0), 1) #vline
        cv2.line(image,(point.x_pos * scale + crosshair_size, point.y_pos * scale) ,\
        (point.x_pos * scale - crosshair_size, point.y_pos * scale), (0, 0, 0), 1) #hline

        # Display the temperature text
        # FIXME: Implement arguments as private class constants
        cv2.putText(image,
                    f'{point.temperature} C',
                    (point.x_pos * scale + 10, point.y_pos * scale - 10),
                    self.font,
                    0.45,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA)
        cv2.putText(image,
                    f'{point.temperature} C',
                    (point.x_pos * scale + 10, point.y_pos * scale - 10),
                    self.font,
                    0.45,
                    (0, 255, 255),
                    1,cv2.LINE_AA)

        # Display optional text
        if point_name:
            cv2.putText(image,
                        f'{point_name}',
                        (point.x_pos * scale + 10, point.y_pos * scale - 25),
                        self.font,
                        0.45,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA)
            cv2.putText(image,
                        f'{point_name}',
                        (point.x_pos * scale + 10, point.y_pos * scale - 25),
                        self.font,
                        0.45,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA)


    def _start_recording(self):
        """ Starts a video recording of the window contents

        Returns:
           VideoWriter : A handle to the video file, that can be used for appending frames.
        """
        now = time.strftime("%Y%m%d--%H%M%S")
        #do NOT use mp4 here, it is flakey!
        video_handle = cv2.VideoWriter(now + 'output.avi',
                                        cv2.VideoWriter_fourcc(*'XVID'),
                                        25,
                                        (self.thermal_camera.scaled_width,
                                        self.thermal_camera.scaled_height))
        return video_handle

    def _snapshot(self, image):
        """ Creates a snapshot of the current contents of
            the window and saves it in the same folder as the program

        Args:
            image (UMat): The image matrix

        Returns:
            string: Capture time.
        """
        #I would put colons in here, but it Win throws a fit if you try and open them!
        now = time.strftime("%Y%m%d-%H%M%S")
        snaptime = time.strftime("%H:%M:%S")
        cv2.imwrite(f"TC{now}.png", image)
        return snaptime


    def _create_window(self):
        """ Creates an OpenCV window for displaying the camera stream """

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME,
                        self.thermal_camera.scaled_width,
                        self.thermal_camera.scaled_height)
        cv2.setMouseCallback(self.WINDOW_NAME, self._handle_mouse_input, param=self)

@click.command()
@click.option("--device", default=0, help="Video Device number e.g. 0, use v4l2-ctl --list-devices")
@click.option("--scale", default=3, help="The scaler to scale the image with, default 3.")
@click.option("--alpha", default=1.0, help="Contrast control 1.0-3.0")
@click.option("--colormap",
            type=click.Choice(['Jet',
                            'Hot',
                            'Magma',
                            'Inferno',
                            'Plasma',
                            'Bone',
                            'Spring',
                            'Autumn',
                            'Viridis',
                            'Parula',
                            'InvRainbow'], case_sensitive=True),
            default="Jet", help="Colormap to use.")
def main(device, scale, alpha, colormap):
    """ Main entry point for the application

    Args:
        device (int): The video device number, check /dev/videoX
        scale (int): The scaling factor to apply to the read image
        alpha (float): The contrast value to apply to the camera image
        colormap (string): The colormap to apply to the camera image
    """
    camera = ThermalCamera(device, scale, alpha, colormap)
    ThermalApp(camera)

def usage():
    """ Prints the author information and the keyboard shortcuts. """

    # FIXME: Implemennt proper printing of help

    print(r"""
    Les Wright 21 June 2023
    https://youtube.com/leslaboratory
    A Python program to read, parse and display thermal data from the Topdon TC001 Thermal camera!

    Tested on Debian all features are working correctly
    This will work on the Pi However a number of workarounds are implemented!
    Seemingly there are bugs in the compiled version of cv2 that ships with the Pi!

    Key Bindings:

    a z: Increase/Decrease Blur
    s x: Floating High and Low Temp Label Threshold
    d c: Change Interpolated scale Note: This will not change the window size on the Pi
    f v: Contrast
    e w: Fullscreen Windowed (note going back to windowed does not seem to work on the Pi!)
    r t: Record and Stop
    p : Snapshot
    m : Cycle through ColorMaps
    h : Toggle HUD
    q : Quit
    """)



if __name__ == "__main__":
    main() # pylint: disable=E1120

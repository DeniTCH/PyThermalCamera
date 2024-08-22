#!/usr/bin/env python3
'''
Les Wright 21 June 2023,
https://youtube.com/leslaboratory

Refactored by @DeniTCH August 2024

A Python program to read, parse and display thermal data from the Topdon TC001 or IniRay P2Pro Thermal camera!
'''

import cv2
import numpy as np
import argparse
import time
import io
import click

class ThermalApp:

    COLORMAPS = {
        'Jet': 0,
        'Hot': 1,
        'Magma': 2,
        'Inferno': 3,
        'Plasma': 4,
        'Bone': 5,
        'Spring': 6,
        'Autumn': 7,
        'Viridis': 8,
        'Parula': 9,
        'InvRainbow': 10
    }

    def __init__(self, device, scale, camera, alpha, colormap):
        #We need to know if we are running on the Pi, because openCV behaves a little oddly on all the builds!
        #https://raspberrypi.stackexchange.com/questions/5100/detect-that-a-python-program-is-running-on-the-pi    		
        self.is_pi = self._is_raspberry_pi()

        # Initialize the video stream
        self.cap = cv2.VideoCapture(f'/dev/video{device}', cv2.CAP_V4L)

        #pull in the video but do NOT automatically convert to RGB, else it breaks the temperature data!
        #https://stackoverflow.com/questions/63108721/opencv-setting-videocap-property-to-cap-prop-convert-rgb-generates-weird-boolean
        if self.is_pi:
            self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
        else:
            self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0) # For some systems 0.0 need to be replaced by a boolean

        # Define the settings
        self.sensor_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #Sensor width
        self.sensor_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/2) #sensor height - only half, the other half is used for thermal data
        self.scale = scale
        self.scaled_width = self.sensor_width * self.scale
        self.scaled_height = self.sensor_height * self.scale
        self.alpha = alpha
        self.colormap = self.COLORMAPS[colormap]
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fullscreen = False
        
        self.blur_radius = 0
        self.threshold = 2
        self.hud = True
        self.recording = False
        self.elapsed_time = "00:00:00"
        self.snaptime = "None"

        # GUI texts
        self.gui_colormap_text = colormap

        # Create the GUI window
        self._create_window()

        self._start_capture()

    def _start_capture(self):
        while(self.cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = self.cap.read()

            # Split the frame into image data and thermal data
            imdata, thdata = np.array_split(frame, 2)

            #now parse the data from the bottom frame and convert to temp!
            #https://www.eevblog.com/forum/thermal-imaging/infiray-and-their-p2-pro-discussion/200/
            #Huge props to LeoDJ for figuring out how the data is stored and how to compute temp from it.
            #grab data from the center pixel...

            # Extract different temperatures
            self.center_temp = self._extract_center_temp(thdata)
            self.max_temp_pos, self.max_temp = self._extract_max_temp(thdata)
            self.min_temp_pos, self.min_temp = self._extract_min_temp(thdata)
            self.avg_temp = self._extract_avg_temp(thdata)

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

            # Draw the GUI elements and temperature markers
            self._draw_gui(heatmap)

            # Finally, display the image
            cv2.imshow('Thermal', heatmap)

            # If we are recording
            if self.recording:
                self._handle_recording(heatmap)

            self._handle_input(heatmap)

    def _handle_input(self, image):
        key = cv2.waitKey(1)

        if key == ord('a'): #Increase blur radius
            self.blur_radius += 1
        if key == ord('z'): #Decrease blur radius
            self.blur_radius -= 1
            if self.blur_radius <= 0:
                self.blur_radius = 0

        if key == ord('s'): #Increase threshold
            self.threshold += 1
        if key == ord('x'): #Decrease threashold
            self.threshold -= 1
            if self.threshold <= 0:
                self.threshold = 0

        if key == ord('d'): #Increase scale
            self.scale += 1
            if self.scale >= 5:
                self.scale = 5
            self.scaled_width = self.sensor_width * self.scale
            self.scaled_height = self.sensor_height * self.scale
            if not self.fullscreen and not self.is_pi:
                cv2.resizeWindow('Thermal', self.scaled_width, self.scaled_height)
        if key == ord('c'): #Decrease self.scale
            self.scale -= 1
            if self.scale <= 1:
                self.scale = 1
            self.scaled_width = self.sensor_width * self.scale
            self.scaled_height = self.sensor_height * self.scale
            if not self.fullscreen and not self.is_pi:
                cv2.resizeWindow('Thermal', self.scaled_width, self.scaled_height)

        if key == ord('q'): #enable fullscreen
            self.fullscreen = True
            cv2.namedWindow('Thermal', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Thermal', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        if key == ord('e'): #disable fullscreen
            self.fullscreen = False
            cv2.namedWindow('Thermal', cv2.WINDOW_GUI_NORMAL)
            cv2.setWindowProperty('Thermal', cv2.WND_PROP_AUTOSIZE,cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow('Thermal', self.scaled_width, self.scaled_height)

        if key == ord('f'): #contrast+
            self.alpha += 0.1
            self.alpha = round(self.alpha, 1)#fix round error
            if self.alpha >= 3.0:
                self.alpha = 3.0
        if key == ord('v'): #contrast-
            self.alpha -= 0.1
            self.alpha = round(self.alpha, 1)#fix round error
            if self.alpha <= 0:
                self.alpha = 0.0


        if key == ord('h'):
            if self.hud:
                self.hud=False
            else:
                self.hud=True

        if key == ord('m'): #m to cycle through color maps
            self.colormap += 1
            if self.colormap == 11:
                self.colormap = 0

        if key == ord('r') and not self.recording: #r to start reording
            self.video_handle = self._start_capture()
            self.recording = True
            self.recording_start_time = time.time()
        if key == ord('t'): #f to finish reording
            self.recording = False
            self.elapsed_time = "00:00:00"

        if key == ord('p'): #f to finish reording
            self.snaptime = self._snapshot(image)

        if key == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()

    def _handle_recording(self, image):
            elapsed = (time.time() - self.recording_start_time)
            elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed)) 
            #print(elapsed)
            self.video_handle.write(image)

    def _draw_gui(self, image):
            # Draw crosshairs
            self._draw_crosshairs(image)

            # Show temperature
            self._show_temperature(image, self.center_temp)

            # Show hud
            if self.hud:
                self._draw_hud(image)

            # Display floating max temperature
            if self.max_temp > self.avg_temp + self.threshold:
                self._draw_circle(image, self.max_temp_pos, (0, 0, 255), self.max_temp)

            # Display floating min temperature
            if self.min_temp < self.avg_temp - self.threshold:
                self._draw_circle(image, self.min_temp_pos, (255, 0, 0), self.min_temp)

    def _draw_circle(self, image, position, color, temp):
        col = position[0]
        row = position[1]

        cv2.circle(image, (row*self.scale, col*self.scale), 5, (0,0,0), 2)
        cv2.circle(image, (row*self.scale, col*self.scale), 5, color, -1)
        cv2.putText(image,str(temp)+' C', ((row*self.scale)+10, (col*self.scale)+5),\
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,0,0), 2, cv2.LINE_AA)
        cv2.putText(image,str(temp)+' C', ((row*self.scale)+10, (col*self.scale)+5),\
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 255, 255), 1, cv2.LINE_AA)        

    def _draw_hud(self, image):
        # Display black box for our data
        cv2.rectangle(image, (0, 0), (160, 120), (0,0,0), -1)
        # Put text in the box
        cv2.putText(image, f'Avg Temp: {self.avg_temp} C', (10, 14),\
        self.font, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(image, f'Label Threshold: {self.threshold} C', (10, 28),\
        self.font, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(image, f'Colormap: {self.gui_colormap_text}', (10, 42),\
        self.font, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(image, f'Blur: {self.blur_radius}', (10, 56),\
        self.font, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(image, f'Scaling: {self.scale}', (10, 70),\
        self.font, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(image, f'Contrast: {self.alpha}', (10, 84),\
        self.font, 0.4,(0, 255, 255), 1, cv2.LINE_AA)


        cv2.putText(image,f'Snapshot: {self.snaptime}', (10, 98),\
        self.font, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

        if not self.recording:
            cv2.putText(image, f'Recording: {self.elapsed_time}', (10, 112),\
            self.font, 0.4,(200, 200, 200), 1, cv2.LINE_AA)
        else:
            cv2.putText(image, f'Recording: {self.elapsed_time}', (10, 112),\
            self.font, 0.4,(40, 40, 255), 1, cv2.LINE_AA)


    def _draw_crosshairs(self, image):
        cv2.line(image,(int(self.scaled_width/2),int(self.scaled_height/2)+20),\
        (int(self.scaled_width/2),int(self.scaled_height/2)-20),(255,255,255),2) #vline
        cv2.line(image,(int(self.scaled_width/2)+20,int(self.scaled_height/2)),\
        (int(self.scaled_width/2)-20,int(self.scaled_height/2)),(255,255,255),2) #hline

        cv2.line(image,(int(self.scaled_width/2),int(self.scaled_height/2)+20),\
        (int(self.scaled_width/2),int(self.scaled_height/2)-20),(0,0,0),1) #vline
        cv2.line(image,(int(self.scaled_width/2)+20,int(self.scaled_height/2)),\
        (int(self.scaled_width/2)-20,int(self.scaled_height/2)),(0,0,0),1) #hline        

    def _show_temperature(self, image, temp):
        cv2.putText(image, f'{temp} C', (int(self.scaled_width/2)+10, int(self.scaled_height/2)-10),\
        self.font, 0.45,(0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'{temp} C', (int(self.scaled_width/2)+10, int(self.scaled_height/2)-10),\
        self.font, 0.45,(0, 255, 255), 1, cv2.LINE_AA)

    def _apply_colormap(self, image):
        #apply colormap
        if self.colormap == 0:
            heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)
            self.gui_colormap_text = 'Jet'
        if self.colormap == 1:
            heatmap = cv2.applyColorMap(image, cv2.COLORMAP_HOT)
            self.gui_colormap_text = 'Hot'
        if self.colormap == 2:
            heatmap = cv2.applyColorMap(image, cv2.COLORMAP_MAGMA)
            self.gui_colormap_text = 'Magma'
        if self.colormap == 3:
            heatmap = cv2.applyColorMap(image, cv2.COLORMAP_INFERNO)
            self.gui_colormap_text = 'Inferno'
        if self.colormap == 4:
            heatmap = cv2.applyColorMap(image, cv2.COLORMAP_PLASMA)
            self.gui_colormap_text = 'Plasma'
        if self.colormap == 5:
            heatmap = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
            self.gui_colormap_text = 'Bone'
        if self.colormap == 6:
            heatmap = cv2.applyColorMap(image, cv2.COLORMAP_SPRING)
            self.gui_colormap_text = 'Spring'
        if self.colormap == 7:
            heatmap = cv2.applyColorMap(image, cv2.COLORMAP_AUTUMN)
            self.gui_colormap_text = 'Autumn'
        if self.colormap == 8:
            heatmap = cv2.applyColorMap(image, cv2.COLORMAP_VIRIDIS)
            self.gui_colormap_text = 'Viridis'
        if self.colormap == 9:
            heatmap = cv2.applyColorMap(image, cv2.COLORMAP_PARULA)
            self.gui_colormap_text = 'Parula'
        if self.colormap == 10:
            heatmap = cv2.applyColorMap(image, cv2.COLORMAP_RAINBOW)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            self.gui_colormap_text = 'Inv Rainbow'      

        return heatmap  

    def _extract_center_temp(self, thdata):
        center_x = int(self.sensor_width/2)
        center_y = int(self.sensor_height/2)

        hi = int(thdata[center_x][center_y][0])
        lo = int(thdata[center_x][center_y][1])
        #print(hi,lo)
        lo = lo*256
        rawtemp = hi+lo
        #print(rawtemp)
        temp = (rawtemp/64.0)-273.15
        temp = round(temp,2)

        return temp

    def _extract_max_temp(self, thdata):
        #find the max temperature in the frame
        lomax = int(thdata[...,1].max())
        posmax = int(thdata[...,1].argmax())
        #since argmax returns a linear index, convert back to row and col
        mcol,mrow = divmod(posmax, self.sensor_width)
        himax = int(thdata[mcol][mrow][0])
        lomax = lomax * 256
        maxtemp = himax + lomax
        maxtemp = (maxtemp/64.0) - 273.15
        maxtemp = round(maxtemp, 2)

        return ((mcol, mrow), maxtemp)

    def _extract_min_temp(self, thdata):
        #find the lowest temperature in the frame
        lomin = int(thdata[...,1].min())
        posmin = int(thdata[...,1].argmin())
        #since argmax returns a linear index, convert back to row and col
        lcol,lrow = divmod(posmin, self.sensor_width)
        himin = int(thdata[lcol][lrow][0])
        lomin = lomin * 256
        mintemp = himin+lomin
        mintemp = (mintemp / 64.0) - 273.15
        mintemp = round(mintemp, 2)
        
        return ((lcol, lrow), mintemp)
    
    def _extract_avg_temp(self, thdata):
        #find the average temperature in the frame
        loavg = int(thdata[...,1].mean())
        hiavg = int(thdata[...,0].mean())
        loavg = loavg * 256
        avgtemp = loavg + hiavg
        avgtemp = (avgtemp / 64) - 273.15
        avgtemp = round(avgtemp, 2)        

        return avgtemp

    def _record(self):
        now = time.strftime("%Y%m%d--%H%M%S")
        #do NOT use mp4 here, it is flakey!
        video_handle = cv2.VideoWriter(now+'output.avi', 
                                        cv2.VideoWriter_fourcc(*'XVID'),
                                        25, 
                                        (self.scaled_width, self.scaled_height))
        return video_handle

    def _snapshot(self, image):
        #I would put colons in here, but it Win throws a fit if you try and open them!
        now = time.strftime("%Y%m%d-%H%M%S") 
        snaptime = time.strftime("%H:%M:%S")
        cv2.imwrite("TC001"+now+".png", image)
        return snaptime
         

    def _create_window(self):
        cv2.namedWindow('Thermal', cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow('Thermal', self.scaled_width, self.scaled_height)


    def _is_raspberry_pi(self):
        try:
            with io.open('/sys/firmware/devicetree/base/model', 'r') as m:
                if 'raspberry pi' in m.read().lower(): 
                    return True
        except Exception:
            pass
        return False    		
            

@click.command()
@click.option("--device", default=0, help="Video Device number e.g. 0, use v4l2-ctl --list-devices")
@click.option("--scale", default=3, help="The scaler to scale the image with, default 3.")
@click.option("--camera", 
            type=click.Choice(['P2Pro', 'TC001'], case_sensitive=True),
            default="P2Pro", help="Model of the camera to use.")
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
def main(device, scale, camera, alpha, colormap):
    ta = ThermalApp(device, scale, camera, alpha, colormap)

@click.command()
def usage():
    click.echo(click.get_help())
    
    print(f"""
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
    q w: Fullscreen Windowed (note going back to windowed does not seem to work on the Pi!)
    r t: Record and Stop
    p : Snapshot
    m : Cycle through ColorMaps
    h : Toggle HUD
    """)



if __name__ == "__main__":
    main()
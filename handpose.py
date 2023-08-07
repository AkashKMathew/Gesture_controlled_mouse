from tkinter import filedialog
import cv2 as cv
import numpy as np
import handgestures as hg
import mouse
from sympy import symbols, Eq, solve
import screeninfo
import time
import tkinter
import customtkinter
import threading
import traceback
import pyautogui
import os
from datetime import datetime
import sys
from PIL import Image,ImageTk
from os import walk
import customtkinter as ctk


cap = cv.VideoCapture(0)

WIDTH = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
RESIZE_BY = 1

enable_sys = False

traversed_points = []
last_click = -1
last_active = -1
drag_enabled = False
last_coordinates = [0,0]
screen_shot_frames = 0 # find how many frames with ss
without_ss_frames = 0 # find how many frames without ss

screen_shot_folder = "screenshots" # current folder as default

next_frame = None

customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("blue") 
viewport_scaling = 1.18


last_key_press = 0
key_press_delay = 1.5

cursor = 1 # 1 for normal cursor/ 0 for presentation



def callback(result, frame):
    global traversed_points, next_frame, last_click, last_active, drag_enabled, last_coordinates,screen_shot_frames,without_ss_frames, last_key_press
    #hand index
    index = -1
    isActive = False
    initiateClick = False
    drag = False
    scroll_up = False
    scroll_down = False

    try:
    #get handedness from result
        if result.hand_landmarks: #get the landmarks from the active hand
            for i,handLms in enumerate(result.hand_landmarks):
                if result.handedness[i][0].display_name == "Right" and handLms[2].x < handLms[17].x:
                    click_pinch_dist = GestureHandler.getDistance(result.hand_landmarks[i][4], result.hand_landmarks[i][12]) # thumb and middle finger
                    drag_pinch_dist = GestureHandler.getDistance(result.hand_landmarks[i][4], result.hand_landmarks[i][8]) # thumb and index finger
                    # ref_dist = GestureHandler.getDistance(result.hand_landmarks[i][4], result.hand_landmarks[i][7])
                    thumb_size = GestureHandler.getDistance(result.hand_landmarks[i][4], result.hand_landmarks[i][3])  

                    if result.gestures[i][0].category_name != "Closed_Fist" and screen_shot_frames > 0: # handle frames inbetween where gesture detection goes wrong
                        without_ss_frames += 1
                        if without_ss_frames == 15:
                            screen_shot_frames = 0
                            without_ss_frames = 0

                    if result.gestures[i][0].category_name == "Open_Palm" and cursor:
                        cv.putText(frame, "Active", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        isActive = True
                        index = i
                    elif result.gestures[i][0].category_name == "Closed_Fist" and False:   # put apple vision pro to shame
                        cv.putText(frame, "Click", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        initiateClick = True
                        index = i
                    elif result.gestures[i][0].category_name == "Thumb_Up" and cursor:
                        cv.putText(frame, "Scroll Up", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        scroll_up = True
                        index = i
                    elif result.gestures[i][0].category_name == "Thumb_Down" and cursor:
                        cv.putText(frame, "Scroll Down", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        index = i
                        scroll_down = True  
                    elif result.gestures[i][0].category_name == "Closed_Fist": # screen shot gesture
                        screen_shot_frames += 1
                        if screen_shot_frames == 60:
                            print("Screenshoted")
                            screen_shot_frames = 0
                            without_ss_frames = 0

                            img = pyautogui.screenshot()
                            img = cv.cvtColor(np.array(img),cv.COLOR_RGB2BGR)

                            # Get current datetime
                            current_datetime = datetime.now()

                            # Get formatted datetime string
                            datetime_string = current_datetime.strftime("%Y-%m-%d %H-%M-%S")

                            # Save screenshot with datetime in the filename
                            datetime_string = f'{datetime_string}.png'
                            cv.imwrite(os.path.join(screen_shot_folder,datetime_string),img)

                        
                    elif drag_pinch_dist < thumb_size*0.85 and click_pinch_dist > thumb_size * 1.5 and cursor:
                        cv.putText(frame, "Drag", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        index = i
                        drag = True
                    elif click_pinch_dist < thumb_size*0.85 and cursor: # handle click using apple pro vision big deal gesture
                        '''landmarks ->  8(index)  as ref
                            landmarks -> 4(index), 12 as click                  
                        check if the index finger and thumb are close enough 
                        use depth of wrist to determine the closeness of the fingers
                        if yes, initiate click
                        ''' 
                        initiateClick = True
                        index = i

                        

                elif result.handedness[i][0].display_name == "Left":
                    if result.gestures[i][0].category_name == "Thumb_Up":
                        cv.putText(frame, "Scroll Up", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        scroll_up = True
                        index = i
                    elif result.gestures[i][0].category_name == "Thumb_Down":
                        cv.putText(frame, "Scroll Down", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        index = i
                        scroll_down = True

                if not cursor: # for presentation control
                    ''' keypoints 4, 3, and 2 should be in a straightline, and 
                    keypoints 5, 6, 7, and 8 should also be in a straightline,
                    keypoints 9, 10, 11, and 12 should be in a straightline,
                    first set should be perpendicular to the second for a valid gesture
                    '''
                    thumb_is_straight = GestureHandler.isStraightLine([result.hand_landmarks[i][4], result.hand_landmarks[i][3], result.hand_landmarks[i][2]])
                    index_is_straight = GestureHandler.isStraightLine([result.hand_landmarks[i][5], result.hand_landmarks[i][6], result.hand_landmarks[i][7], result.hand_landmarks[i][8]])
                    middle_is_straight = GestureHandler.isStraightLine([result.hand_landmarks[i][9], result.hand_landmarks[i][10], result.hand_landmarks[i][11], result.hand_landmarks[i][12]])
                    # print (thumb_is_straight, index_is_straight, middle_is_straight)
                    if thumb_is_straight and index_is_straight and middle_is_straight:
                        # get angle between the two lines
                        # print(GestureHandler.getAngle([result.hand_landmarks[i][4],result.hand_landmarks[i][2]],[result.hand_landmarks[i][5],result.hand_landmarks[i][8]]))
                        # print(GestureHandler.getAngle([result.hand_landmarks[i][4],result.hand_landmarks[i][2]],[result.hand_landmarks[i][9],result.hand_landmarks[i][12]]))
                        if GestureHandler.getAngle([result.hand_landmarks[i][4],result.hand_landmarks[i][2]],[result.hand_landmarks[i][5],result.hand_landmarks[i][8]]) > 70 \
                            and GestureHandler.getAngle([result.hand_landmarks[i][4],result.hand_landmarks[i][2]],[result.hand_landmarks[i][9],result.hand_landmarks[i][12]]) >70:
                            # check which direction the hand is pointing to
                            if result.hand_landmarks[i][2].x < result.hand_landmarks[i][8].x and result.hand_landmarks[i][2].x < result.hand_landmarks[i][12].x and \
                                time.time() - last_key_press > key_press_delay:
                                last_key_press = time.time()
                                cv.putText(frame, "Next", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                pyautogui.press('right')
                                print("Next")
                            elif result.hand_landmarks[i][2].x > result.hand_landmarks[i][8].x and result.hand_landmarks[i][2].x > result.hand_landmarks[i][12].x and \
                                time.time() - last_key_press > key_press_delay:
                                last_key_press = time.time()
                                cv.putText(frame, "Previous", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                pyautogui.press('left')
                                print("Previous")
                    
                    

        if isActive or drag: # if active, draw the path
            if drag or isActive:
                landmarks = [result.hand_landmarks[index][2], result.hand_landmarks[index][5], result.hand_landmarks[index][9], result.hand_landmarks[index][13], result.hand_landmarks[index][17], result.hand_landmarks[index][0]]
                coordinates =  GestureHandler.getAvg(landmarks)
                last_coordinates = traversed_points[-1] if len(traversed_points) > 0 else [int(coordinates[0] * WIDTH), int(coordinates[1] * HEIGHT)]
                if drag and not drag_enabled:
                    drag_enabled = True
                elif not drag and drag_enabled:
                    drag_enabled = False
            last_active = time.time()

            # set traversed points as the average of landmarks 2,5.9.13,17 and 0
            # print([int(coordinates[0] * WIDTH), int(coordinates[1] * HEIGHT)])
            traversed_points.append([int(coordinates[0] * WIDTH), int(coordinates[1] * HEIGHT)])
            if len(traversed_points) > 10:
                traversed_points.pop(0)
            # for i in range(len(traversed_points)-1):
            #     cv.line(frame, (traversed_points[i][0],traversed_points[i][1]),( traversed_points[i+1][0], traversed_points[i+1][1]),(255,0,0), 5)
            # cv.putText(frame, "Active", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # handleMouseDrag(traversed_points[-2], traversed_points[-1])
            # print(traversed_points)
            if len(traversed_points) > 1:
                simpleMouseDrag(traversed_points[-2].copy(), traversed_points[-1].copy(),drag=drag_enabled)
        elif scroll_up:
            mouse.wheel(10)
        elif scroll_down:
            mouse.wheel(-10)
        elif initiateClick and time.time() - last_click > 1:
            mouse.click('left')
            last_click = time.time()
        elif not isActive and time.time() - last_active > 5: # if not active, clear the prev path
            traversed_points = []
        # print(traversed_points,'\n')
    except Exception as e:
        traceback.print_exc()
    finally:
        frame = GestureHandler.drawLandMarks(frame,  result)
        next_frame = frame

GestureHandler = hg.GestureHandler(callback=callback)


def handleMouseDrag(prev_point, curr_point): 
    '''SOLUTION USING TRIGONOMETRY'''
    #TODO
    curr_mouse_pos = mouse.get_position()
    slope = (curr_point[1] - prev_point[1]) / (curr_point[0] - prev_point[0]) # slope of the line
    magnitude = np.sqrt((curr_point[1] - prev_point[1])**2 + (curr_point[0] - prev_point[0])**2) # distance between the two points

    next_x, next_y = -1,-1
    # solve for the next mouse point
    x = symbols('x')
    solve_for_x = Eq((x - curr_mouse_pos[0])**2 + (slope*(x - curr_mouse_pos[0]) + curr_mouse_pos[1])**2, magnitude**2)
    solutions = solve(solve_for_x)
    
    # select appropriate solution
    for soln in solutions:
        if soln > curr_mouse_pos[0] and soln < WIDTH:
            next_x = soln
            break

    next_y = slope*(next_x - curr_mouse_pos[0]) + curr_mouse_pos[1]
    mouse.move(next_x, next_y, absolute=True, duration=0.1)

def simpleMouseDrag(prev_point, curr_point, drag = False):
    SCREEN_OFFSET = 2
    POINT_OFFSET = viewport_scaling 
    # prev_point = np.array(prev_point)/SCREEN_OFFSET
    # curr_point = np.array(curr_point)/SCREEN_OFFSET
    screenWidth, screenHeight = screeninfo.get_monitors()[0].width, screeninfo.get_monitors()[0].height
    prev_point[0] = max(prev_point[0] - WIDTH/2.5,0)*POINT_OFFSET
    prev_point[1] = max(prev_point[1] - HEIGHT/2.5,0)*POINT_OFFSET
    curr_point[0] = max(curr_point[0] - WIDTH/2.5,0)*POINT_OFFSET
    curr_point[1] = max(curr_point[1] - HEIGHT/2.5,0)*POINT_OFFSET
    #map the points to the screen size
    adjust = (1 -SCREEN_OFFSET)/2 * 0
    prev_point = (int(((prev_point[0] * screenWidth / WIDTH)*SCREEN_OFFSET) + screenWidth*adjust), int(((prev_point[1] * screenHeight/ HEIGHT)*SCREEN_OFFSET + screenHeight*adjust)))
    curr_point = (int(((curr_point[0] * screenWidth   / WIDTH)*SCREEN_OFFSET) + screenWidth*adjust), int(((curr_point[1] * screenHeight / HEIGHT)*SCREEN_OFFSET+ screenHeight*adjust)))

    # print("dragging from ", prev_point, " to ", curr_point, "") # debug
    #move the mouse
    if drag:
        mouse.drag(prev_point[0], prev_point[1],curr_point[0], curr_point[1], absolute=True, duration=0)
    else:
        mouse.move(curr_point[0], curr_point[1], absolute=True, duration=0)

def setupWindow():
    class AnimatedButton(ctk.CTkButton):
        def __init__(self, parent, dark_path):
            
        # 	# animation logic setup
            self.frames = self.import_folders( dark_path)
            # print(self.frames)
            self.frame_index = 0
            self.animation_length = len(self.frames) - 1
            self.animation_status = ctk.StringVar(value = 'start')

            self.animation_status.trace('w', self.animate)

            super().__init__(
                master = parent, 
                text = 'A animated button',
                image=self.frames[self.frame_index],
                compound = 'top',)
            # self.pack(expand = True)

        # def infinite_animate(self):
        # 	self.frame_index += 1
        # 	self.frame_index = 0 if self.frame_index > self.animation_length else self.frame_index 
        # 	self.configure(image = self.frames[self.frame_index])
        # 	self.after(20, self.infinite_animate)

        def import_folders(self, dark_path):
            
            # image_paths = []
            
            for _, __, image_data in walk(dark_path):
                # print(image_data)
                sorted_data = sorted(
                    image_data, 
                    key = lambda item: int(item.split('.')[0][-2:]))
                # print(sorted_data)
                full_path_data = [dark_path + '/' + item for item in sorted_data]
                # image_paths.append(full_path_data)
            # image_paths = zip(*image_paths)
            # print(image_paths)
            
            ctk_images = []
            for image_path in full_path_data:
                # print(image_path)
                dark_image = Image.open(image_path)
                dark_image = dark_image.resize((80, 80))
                ctk_image = ImageTk.PhotoImage(dark_image)
                ctk_images.append(ctk_image)
            # print(ctk_images)
            return ctk_images

        def trigger_animation(self):
            if self.animation_status.get() == 'start':
                self.frame_index = 0
                self.animation_status.set('forward')
            if self.animation_status.get() == 'end':
                self.frame_index = self.animation_length
                self.animation_status.set('backward')

        def animate(self, *args):
            # print(self.frame_index)
            if self.animation_status.get() == 'forward':
                self.frame_index += 1
                self.configure(image = self.frames[self.frame_index])

                if self.frame_index < self.animation_length:
                    self.after(20, self.animate)
                else:
                    self.animation_status.set('end')

            if self.animation_status.get() == 'backward':
                self.frame_index -= 1
                self.configure(image = self.frames[self.frame_index])

                if self.frame_index > 0:
                    self.after(20, self.animate)
                else:
                    self.animation_status.set('start')


    def combine_funcs(*funcs):
  
        # this function will call the passed functions
        # with the arguments that are passed to the functions
        def inner_combined_func(*args, **kwargs):
            for f in funcs:
    
                # Calling functions with arguments, if any
                f(*args, **kwargs)
    
        # returning the reference of inner_combined_func
        # this reference will have the called result of all
        # the functions that are passed to the combined_funcs
        return inner_combined_func

    app = customtkinter.CTk()
    app.title("FingerFlex")
    app.geometry("600x400")
    # app.iconbitmap("assets/folder.png")
    app.grid_columnconfigure(0, weight=1)
    app.grid_columnconfigure(1, weight=1)
    app.grid_columnconfigure(2, weight=1)
    app.grid_columnconfigure(3, weight=1)
    app.grid_rowconfigure(0, weight=1)
    app.grid_rowconfigure(1, weight=1)
    app.grid_rowconfigure(2, weight=1)
    image = Image.open("assets/bg.jpg")
    image = image.resize((600,400))
    bg_image = ImageTk.PhotoImage(image)

    bg_label = customtkinter.CTkLabel(app,image=bg_image,text="")
    bg_label.place(x=0,y=0,relwidth=1,relheight=1)

    switch = AnimatedButton(app, 'vision')
    switch.configure(text="Enable Vision",height=100,width=130, command=combine_funcs(toggleVision,switch.trigger_animation),anchor='center')
    switch.grid(row=0, column=0, rowspan=2, columnspan=2)

    def toggleMode():
        global cursor
        cursor = 1 -cursor
        mode_switch.configure(text="Cursor Mode" if  cursor else "Presentation Mode")

    mode_switch = AnimatedButton(app,'cursor')
    mode_switch.configure(text="Cursor Mode" if  cursor else "Presentation Mode",height=100,width=130,command=combine_funcs(toggleMode,mode_switch.trigger_animation),anchor='center')
    mode_switch.grid(row=1,column=0,rowspan=2,columnspan=2)
    
    # Slider current values
    # sense_value = customtkinter.DoubleVar()
    viewport_value = customtkinter.DoubleVar(value=viewport_scaling)

    def get_current_value(arg):
        global viewport_scaling
        if arg == 2:
            viewport_scaling  = round(float(viewport_value.get()), 2)
            return viewport_scaling 

    def slider_changed(arg):
        if arg == 2:
            value_label2.configure(text=get_current_value(2))

    

    
    def screenshot():
        img = pyautogui.screenshot()
        img = cv.cvtColor(np.array(img),cv.COLOR_RGB2BGR)
        # Get current datetime
        current_datetime = datetime.now()
        # Get formatted datetime string
        datetime_string = current_datetime.strftime("%Y-%m-%d %H-%M-%S")
        # Save screenshot with datetime in the filename
        datetime_string = f'{datetime_string}.png'
        cv.imwrite(os.path.join(screen_shot_folder,datetime_string),img)
    SSbutton = AnimatedButton(app, 'ss')
    SSbutton.configure(text="Take Screenshot",height=100,width=100, command=combine_funcs(screenshot,SSbutton.trigger_animation))
    SSbutton.grid(
        column=2,
        row=0,
        columnspan=2,
        sticky='s',
    )
    
    def folder():
        filename = filedialog.askopenfilename(initialdir = "./screenshots",
                                          title = "Select a File",
                                          filetypes = (
                                                        ("all files",
                                                        "*.*"),
                                                        ("Text files",
                                                        "*.txt*"),
                                                       ))
        image_path = filename  # Replace with the actual path to your image file
        image = Image.open(image_path)
        image.show()
        
    folder_img = Image.open("assets/folder.png")
    folder_img = folder_img.resize((20,20))
    SSfolder = customtkinter.CTkButton(app, text="Browse Files",image=ImageTk.PhotoImage(folder_img), command=folder)
    SSfolder.grid(
        column=2,
        row=1,
        columnspan=2,
        sticky='n',
        pady=2,
    )
    view = Image.open("assets/area-graph.png")
    view = view.resize((40,40))
    slider_button = customtkinter.CTkButton(
        app,
        text='Viewport : ',
        bg_color='#1f6aa5',
        height=80,width=80,
        compound='top',
        image=ImageTk.PhotoImage(view),
    )
    slider_button.grid(
        column=2,
        row=1,
        sticky='se',
    )

    slider2 = customtkinter.CTkSlider(master=app,
                                      width=130,
                                      height=16,
                                      border_width=10,
                                      from_ = 0,
                                      to = 5,
                                      command=lambda event: slider_changed(2),
                                      variable=viewport_value,
                                      bg_color='#1f6aa5',
                                      fg_color='white',
                                      button_color='#09004f',
                                      )

    slider2.grid(
        column=2,
        row=2,
        columnspan=2,
        sticky='n',
    )

    # Value label
    value_label2 = customtkinter.CTkButton(
        app,
        text=get_current_value(2),
        height=80,width=50,
        bg_color='#1f6aa5',

    )
    value_label2.grid(
        row=1,
        column=3,
        sticky='sw',
    )

    def on_closing():
        cap.release()  # Release the camera capture
        sys.exit()  # Exit the program

    app.protocol("WM_DELETE_WINDOW", on_closing) 

    app.mainloop()



def toggleVision():
    global enable_sys
    enable_sys = not enable_sys

if __name__ == "__main__":

    if not os.path.exists(screen_shot_folder): # create screen shots folder if not exists
        os.mkdir(screen_shot_folder)

    threading.Thread(target=setupWindow).start()
 
    while True:
        isActive = False # True if the active hand is facing the camera
        ret, frame = cap.read()

        # resize frame
        frame = cv.resize(frame, (int(WIDTH/RESIZE_BY),int( HEIGHT/RESIZE_BY)), interpolation=cv.INTER_AREA)

        frame = cv.flip(frame, 1) # flip the frame horizontally

        if not ret:
            break

        if enable_sys:
            GestureHandler.getLandmarksNGesture(cv.cvtColor(frame,cv.COLOR_BGR2RGB)) # get the landmarks from the active hand
        
        if next_frame is not None:
            cv.imshow("frame", cv.cvtColor(next_frame,cv.COLOR_RGB2BGR)) 
            next_frame = None  

        if cv.waitKey(10) & 0xFF == 27:
            break
        
    cap.release()
    cv.destroyAllWindows()

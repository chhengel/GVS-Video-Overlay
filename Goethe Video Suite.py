import tkinter as tk
from tkinter import messagebox
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from datetime import timedelta
import platform
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrow, Circle, Rectangle

import sys, os
import platform

global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added, timestamp_size
global start_x,start_y,end_x,end_y,scale_length,scale_image,x_shift,y_shift,pos_x,pos_y,lnorm, lstart_x, lstart_y,lend_x
global imageflag       
global addedscale
addedscale=False
current_frame=0
timestamp_added=False
fps=30.
timestamp_size=None
timestamp_angle=None
timestamp_position=(10,10)
end_x=-1
start_x=-1
lend_x=-1
lstart_x=-1
lnorm=-1


class ButtonScript:
    def __init__(self, root, main_menu_callback):
        self.root = root
        root.title("Button-Skript")
        root.geometry("1000x800")
       

        self.button = tk.Button(root, text="Button", command=self.show_message)
        self.button.pack(pady=10)
        

        self.main_menu_callback = main_menu_callback

    def show_message(self):
        messagebox.showinfo("Info", "Gedrückt")
        self.root.destroy()  # Schließe das aktuelle Fenster
        self.main_menu_callback()  # Zeige das Hauptmenü wieder an

class AnotherButtonScript:
    def __init__(self, root, main_menu_callback):
        self.root = root
        root.title("Another Button-Skript")

        self.button = tk.Button(root, text="Button", command=self.show_message)
        self.button.pack(pady=10)

        self.main_menu_callback = main_menu_callback

    def show_message(self):
        messagebox.showinfo("Info", "Nochmal")
        self.root.destroy()  # Schließe das aktuelle Fenster
        self.main_menu_callback()  # Zeige das Hauptmenü wieder an
       

class ScriptSelector:
    global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added, timestamp_size
    global imageflag
    def __init__(self, root):
        self.root = root
        root.title("Goethe-Video-Suite")
        root.geometry('500x810')
        self.t0=t0_time
        self.cap = None
        self.basedir = os.path.dirname(__file__)
        #self.cached_frames=cached_frames
        #self.total_frames=total_frames
        #self.current_frame=current_frame
        #self.fps=fps
        #self.fps_native=fps_native
        #t0_frame=t0_frame
        #t0_time=t0_time
        #self.start_point=start_point
        #self.scale_added=scale_added
        #self.cap=None

        self.bg = tk.PhotoImage(file = os.path.join(self.basedir,"VSBackground4.png"))
        self.bgLabel = tk.Label(root,image = self.bg)
        self.bgLabel.place(x=0,y=0)

        #self.head0 = tk.Label(root,text="")
        #self.head0.pack(pady=50)
        

        self.main_menu_button = tk.Button(root, text="Video öffnen", width=20,height=5, command=self.open_video)
        self.main_menu_button.place(x=150,y=100)

        self.open_image_button = tk.Button(root, text="Image öffnen", width=20, height =5,command=self.open_image)
        self.open_image_button.place(x=150,y=220)

        self.video_slider = tk.Scale(root, from_=0, to=100,resolution=1, orient="horizontal", length=300)
        
        self.video_slider = tk.Scale(root, from_=0, to=100,resolution=1, orient="horizontal", command=self.show_frame, length=300)
        self.video_slider.place(x=100,y=340)

        #self.button_script_button = tk.Button(root, image=bg)
        #self.button_script_button.pack(pady=10)

        #self.Time_Image = tk.PhotoImage(file = os.path.join(self.basedir,"TimeStamp small.png"))

        #self.another_button_script_button = tk.Button(root, text="Timestamp", image=self.Time_Image, compound='top', command=self.run_Timestamp)
        #self.another_button_script_button.pack(padx=10, pady=10,side='top')

        #self.KO_Image = tk.PhotoImage(file = os.path.join(self.basedir,"KO small.png"))

        #self.KO_Sys_button = tk.Button(root,text="KO-System einfügen", image=self.KO_Image, command=self.run_KO_Sys, compound='top')
        #self.KO_Sys_button.pack(padx=10, pady=10, side='top')

        #self.Crop_Image = tk.PhotoImage(file = os.path.join(self.basedir,"Crop small.png"))
 
        #self.Crop_button = tk.Button(root,text="Video zuschneiden und speichern", image=self.Crop_Image, compound='top', command=self.run_Crop)
        #self.Crop_button.pack(padx=10, pady=10,side='top')
        
        root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def show_main_menu(self):
        self.root.deiconify()  # Stelle das Hauptmenü-Fenster wieder her
        #self.scale_added=addedscale
        #print("Hier",self.scale_added)
        if cached_frames is not None:
            #print("I was here")
            frame=cached_frames[current_frame].copy()
            cv2.imshow("Video", frame)
                #cv2.waitKey(1)
            x=self.root.winfo_x()
            y=self.root.winfo_y()
            w=self.root.winfo_width()
            cv2.moveWindow("Video",x+w+30,y)
            cv2.imshow("Video", frame)
            cv2.waitKey(1)
            self.video_slider.set(current_frame)
            self.show_frame(current_frame)
    

    def run_button_script(self):
        root.withdraw()  # Minimiere das Hauptmenü
        button_script_root = tk.Toplevel(root)
        button_script_app = ButtonScript(button_script_root, self.show_main_menu)


    def run_KO_Sys(self):
        if cached_frames is not None:
            #cv2.destroyWindow("Video")
            #cv2.destroyAllWindows()
            root.withdraw()  # Minimiere das Hauptmenü
            KO_Sys_button_script_root = tk.Toplevel(root)
            KO_Sys_button_script_app = KO_Sys(KO_Sys_button_script_root, self.show_main_menu)

    def open_image(self):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
        global imageflag
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added, timestamp_size
        global start_x,start_y,end_x,end_y,scale_length,scale_image,x_shift,y_shift,pos_x,pos_y,lnorm, lstart_x, lstart_y,lend_x

        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.bmp *.jpg")])
        if file_path:
            frame=cv2.imread(file_path)
            height_px = int(0.9*root.winfo_screenheight())
            if frame.shape[0]>height_px and platform.system() != "Darwin":
                    tempimage = self.ResizeWithAspectRatio(frame, height=height_px)
                    frame=tempimage 

            cached_frames = {0:frame.copy()}
            current_frame=0
            total_frames=1
            fps=0
            fps_native=0
            self.first_frame =cached_frames[0].copy()
            imageflag=True
            cv2.imshow("Video", frame)
            x=self.root.winfo_x()
            y=self.root.winfo_y()
            w=self.root.winfo_width()
            w=500
            cv2.moveWindow("Video",x+w+30,y)
            cv2.imshow("Video", frame)
            cv2.waitKey(1)
            self.run_KO_Sys()

    def open_video(self):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
        global imageflag, timestamp_position
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added, timestamp_size
        global start_x,start_y,end_x,end_y,scale_length,scale_image,x_shift,y_shift,pos_x,pos_y,lnorm, lstart_x, lstart_y,lend_x

        #global cached_frames,total_frames
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if file_path:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(file_path)
            if cached_frames is not None:
                cached_frames.clear()
                cached_frames=None
                total_frames=0
                current_frame=0
                fps=30.
                fps_native=30.
                t0_frame=0
                #t0_time=0
                self.t0=0
                start_point=None
                scale_added=False
                imageflag = False
                current_frame=0
                timestamp_added=False
                fps=30.
                timestamp_size=None
                timestamp_position=(10,10)
                end_x=-1
                start_x=-1
                lend_x=-1
                lstart_x=-1
                lnorm=-1           
            width_px = root.winfo_screenwidth()
            height_px = int(0.9*root.winfo_screenheight())
            #print('HEIGHT',height_px) 
            #if self.first_frame is not None:
             #   if self.first_frame.shape[0]>height_px and platform.system() != "Darwin":
              #      tempimage = self.ResizeWithAspectRatio(self.first_frame, height=height_px)
               #     self.first_frame=tempimage
            
            #self.show_first_frame()
            #cv2.waitKey(5000)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            ret,frame = self.cap.read()
            if ret:
                if frame.shape[0]>height_px and platform.system() != "Darwin":
                    tempimage = self.ResizeWithAspectRatio(frame, height=height_px)
                    frame=tempimage 
                cached_frames = {0:frame.copy()}

                text='VIDEO WIRD GELADEN!'
                font = cv2.FONT_HERSHEY_DUPLEX
                #imgtext=self.first_frame
                textsize = cv2.getTextSize(text, font, 3, 2)[0]
                
                textX = int((frame.shape[1] - textsize[0]) / 2)
                textY = int((frame.shape[0] + textsize[1]) / 2 )
                
                #print('TEXT',textX,textY,text)
                
                cv2.putText(frame, text, (textX,textY),font, 3, (0, 0, 255), 2,cv2.LINE_AA)
                cv2.imshow("Video", frame)
                cv2.waitKey(1)
                

                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                #total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.video_slider.config(to=total_frames - 1, resolution=1)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES,1)
                # Lese die Frames ein und speichere sie im Cache-Dictionary
                for i in range(1,total_frames):
                    #self.cap.set(cv2.CAP_PROP_POS_FRAMES,i)
                    ret, frame = self.cap.read()
                    #print(self.total_frames,total_frames,i,self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if frame is not None:
                        if frame.shape[0]>height_px and platform.system() != "Darwin":
                            tempimage = self.ResizeWithAspectRatio(frame, height=height_px)
                            frame=tempimage  
                        cached_frames[i]=frame.copy()
                        if(i%10==0):
                            frame=cached_frames[i].copy()
                            cv2.putText(frame, text, (textX,textY),font, 3, (0, 0, 255), 2,cv2.LINE_AA)
                            cv2.imshow("Video", frame)
                            cv2.waitKey(1)
                    else:
                        total_frames=i-1
                        messagebox.showerror("Fehler beim Einlesen des Videos",f"Abbruch bei Frame {i-1}")
                        self.video_slider.config(to=total_frames - 1, resolution=1)
                        break
                print("cached")
                current_frame=0
                fps=self.cap.get(cv2.CAP_PROP_FPS)
                fps_native=self.cap.get(cv2.CAP_PROP_FPS)
                self.first_frame =cached_frames[0].copy()
                x=self.root.winfo_x()
                y=self.root.winfo_y()
                w=self.root.winfo_width()
                w=500
                cv2.moveWindow("Video",x+w+30,y)
                cv2.imshow("Video", self.first_frame)
                cv2.waitKey(1)
                #cv2.destroyAllWindows()
                #cv2.waitKey(0)
                self.run_KO_Sys()
                #self.show_frame()
                #self.mark_start()

    def show_frame(self,value):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
        #global timestamp_added
        #global cached_frames
        frame=cached_frames[int(value)].copy()
        current_frame=int(value)
        #print("Hallo",self.scale_added)
        if scale_added:
            self.overlay_image(frame)
        if timestamp_added:
            frame1=self.draw_timestamp(frame)
            frame=frame1.copy()
        cv2.imshow("Video",frame)
        #cv2.waitKey(1)

    def draw_timestamp(self, frame):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added, timestamp_size, timestamp_position, timestamp_angle   
        self.t0=t0_time
        #time_diff=(current_frame - t0_frame) / fps+self.t0_rest
        time_diff=current_frame/fps-self.t0
        if time_diff<0:
            negativ=True
            time_diff=-1.*time_diff
        else:
            negativ=False
        ten_minutes=int(time_diff/600.)
        time_diff=time_diff-float(ten_minutes)*600.
        minutes=int(time_diff/60.)
        time_diff=time_diff-float(minutes)*60.
        ten_seconds=int(time_diff/10.)
        time_diff=time_diff-float(ten_seconds)*10.
        seconds=int(time_diff)
        time_diff=time_diff-float(seconds)*1.0
        zehntel=int(time_diff/0.1)
        time_diff=time_diff-float(zehntel)*0.1
        hstel=int(time_diff/.01)
        time_diff=time_diff-float(hstel)*0.01
        tstel=int(time_diff/.001)
        if not negativ:
            timestamp_text = f"Time:  {str(ten_minutes)}{str(minutes)}:{str(ten_seconds)}{str(seconds)}.{str(zehntel)}{str(hstel)}{str(tstel)}"
        else:
            timestamp_text = f"Time: -{str(ten_minutes)}{str(minutes)}:{str(ten_seconds)}{str(seconds)}.{str(zehntel)}{str(hstel)}{str(tstel)}"
        # Berechne die Breite und Höhe des Zeitstempelhintergrunds
        text_scale=timestamp_size
        #text_scale=1.0
        text_width, text_height = cv2.getTextSize(timestamp_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 2)[0]
        background_width = int(text_width * 1.1)
        background_height = int(text_height * 2.5)

        # Zeichne ein Rechteck mit abgerundeten Ecken
        rot_ts_frame = np.zeros_like(frame)
        rot_ts_frame1 = np.zeros_like(frame)
        ts_frame=np.zeros((frame.shape[0]+background_height,frame.shape[1]+background_width,frame.shape[2]), dtype=np.uint8)
        ts_frame1=np.zeros((frame.shape[0]+background_height,frame.shape[1]+background_width,frame.shape[2]), dtype=np.uint8)
        ts_rotation_matrix= cv2.getRotationMatrix2D(timestamp_position,timestamp_angle,1)
        
        #cv2.rectangle(ts_frame, timestamp_position,
         #             (timestamp_position[0] + background_width, timestamp_position[1] + background_height),
          #            (255, 255, 255), -1, cv2.LINE_AA, shift=0)
        cv2.rectangle(ts_frame, timestamp_position,
                      (timestamp_position[0] + background_width, timestamp_position[1] + background_height),
                      (255, 255, 255), -1)
        cv2.rectangle(ts_frame1, (timestamp_position[0]+2,timestamp_position[1]+2),
                      (timestamp_position[0] + background_width-2, timestamp_position[1] + background_height-2),
                      (255, 255, 255), -1)
        cv2.putText(ts_frame, timestamp_text, (timestamp_position[0] + int(10*text_scale), timestamp_position[1] + int(text_scale*40)),
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (1, 1, 1), int(2*text_scale), cv2.LINE_AA)
        rot_ts_frame=cv2.warpAffine(ts_frame,ts_rotation_matrix,(ts_frame.shape[1],ts_frame.shape[0]))[0:frame.shape[0],0:frame.shape[1]]
        rot_ts_frame1=cv2.warpAffine(ts_frame1,ts_rotation_matrix,(ts_frame.shape[1],ts_frame.shape[0]))[0:frame.shape[0],0:frame.shape[1]]
        
        frame[np.where((rot_ts_frame1 != [0,0,0]).all(axis=2))]=[0,0,0]
        frame=cv2.add(frame,rot_ts_frame)
        
        return frame
        
        
        #cv2.rectangle(frame, timestamp_position,
         #             (timestamp_position[0] + background_width, timestamp_position[1] + background_height),
          #            (255, 255, 255), -1, cv2.LINE_AA, shift=0)
        #cv2.putText(frame, timestamp_text, (timestamp_position[0] + int(10*text_scale), timestamp_position[1] + int(text_scale*40)),
         #           cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 0), int(2*text_scale), cv2.LINE_AA)

    def overlay_image(self, background):
        global start_x,start_y,end_x,end_y,scale_length,scale_image,x_shift,y_shift,pos_x,pos_y,lnorm, lstart_x, lstart_y
        #print('x-shift,y-shift',x_shift,y_shift)
        #start_x,start_y=self.start_point
        overlay=scale_image
        ys_min=0
        ys_max=overlay.shape[0]
        xs_min=0
        xs_max=overlay.shape[1]
        
        #alpha=1.-float(self.alpha_slider.get())
        alpha=1.0

        alpha_s = overlay[:, :, 3] / 255.0*alpha
        #alpha_s = 0.5
        alpha_l = 1.0 - alpha_s
        pos_x=start_x
        pos_y=start_y
        y1, y2 = pos_y-y_shift, pos_y-y_shift + overlay.shape[0]
        x1, x2 = pos_x-x_shift, pos_x-x_shift + overlay.shape[1]

        if y1<0:
            ys_min=abs(y1)
            y1=0
        if x1<0:
            xs_min=abs(x1)
            x1=0
        if y2>background.shape[0]:
            ys_max=ys_max-(y2-background.shape[0])
            y2=background.shape[0]
        if x2>background.shape[1]:
            xs_max=xs_max-(x2-background.shape[1])
            x2=background.shape[1]

        for c in range(0, 3):
            background[y1:y2, x1:x2, c] = (alpha_s[ys_min:ys_max,xs_min:xs_max] * overlay[ys_min:ys_max, xs_min:xs_max, c] +
                                           alpha_l[ys_min:ys_max,xs_min:xs_max] * background[y1:y2, x1:x2, c])

    def ResizeWithAspectRatio(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation=inter)

    
    
    def on_closing(self):
        self.root.destroy()
        self.root.quit()

#########################################################################################################
####################################################################################################
####################################################################################################
####### KO-SYSTEM

class KO_Sys:
    global addedscale
    global imageflag
    global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added, timestamp_position
    global start_x,start_y,end_x,end_y,scale_length,scale_image,x_shift,y_shift,pos_x,pos_y,lnorm, lstart_x, lstart_y, scale_markup, lend_x
    def __init__(self, root,main_menu_callback):
        global start_x,start_y,end_x,end_y,scale_length,scale_image,x_shift,y_shift,pos_x,pos_y,lnorm, lstart_x, lstart_y, scale_markup
        self.root = root
        
        #self.cached_frames=cached_frames
        self.basedir = os.path.dirname(__file__)
        self.main_menu_callback = main_menu_callback
        root.title("Bewegungsvideo bearbeiten")
        self.cap = None
        self.first_frame = cached_frames[current_frame].copy()
        self.start_point = start_point
        self.end_point = None
        self.lstart_point = None
        self.lend_point = None
        self.selecting_start_point = True
        self.selecting_lstart_point = True
        self.output_filename = None
        #current_frame = 0
        self.l_selected=False
        self.start_end_selected=False
        self.scale_added=scale_added
        self.invertx=False
        self.start_end_selected_crop=False
        self.selecting_start_point_crop=True
        #end_x=-1
        #start_x=-1
        #lend_x=-1
        #lstart_x=-1
        #lnorm=-1
        #self.cached_frames=None
        self.export_axis=None
        self.timestamp_added=False
        t0_frame=0
        #self.timestamp_position = False
        self.beschleunigung=None
        self.marka1=True
        self.marka2=False
        self.units_per_px=None
        #self.x_shift=x_shift
        #self.y_shift=y_shift
        self.crop_x0=None
        self.crop_y0=None
        self.crop_x1=None
        self.crop_y1=None
        scale_image=None
        self.frame_min=0
        self.frame_max=total_frames
        self.step=1
        self.show=False
        self.first_cal_change=True
        self.lastcal=1.
        self.polar=False
        self.polar_circle=False
        self.selecting_first_polar=True
        self.selecting_second_polar=False
        self.selecting_third_polar=False
        self.selecting_polar_end=False
        self.polar1=None
        self.polar2=None
        self.polar3=None
        self.radius_screen=None
        self.polar_center=None
        self.first_circle=True
        self.selecting_polar_center=True
        self.selecting_polar_radius=False
        self.scale_image_not_rot=False
        self.scale_markup_not_rot=False
        self.rot_angle_bak=False
        self.end_point_bak=False
        self.keyrot_angle=0.
        self.t0_rest=0.
        self.t0=0.
        self.center_coords=None
        self.center_coords_scaled=None
        self.scaling_factor=None
        self.markerdist_in_m=None
        self.markerdist_in_pixels_as_is=None
        
        
        
        root.bind("<Button-1>", self.entries_loose_focus)
        root.bind('<Return>',self.return_was_hit)
        root.bind('<Left>', self.move_scale_left)
        root.bind('<Right>', self.move_scale_right)
        root.bind('<Up>', self.move_scale_up)
        root.bind('<Down>', self.move_scale_down)
        root.bind('<l>',self.rotate_left)
        root.bind('<r>',self.rotate_right)
        root.maxsize(550,820)

############################## MENUE FRAMES #############################################################
############################## FRAME 1 NAVIGATION IM VIDEO ##############################################
        frame1=tk.Frame(root,width=450,borderwidth=2,relief='sunken')
        frame1.grid(row=0,column=0,padx=10,pady=10,sticky='ew')
        frame1.columnconfigure(0,weight=10)
        frame1.columnconfigure(1, weight=1)
        frame1.columnconfigure(2,weight=1)

        self.video_slider = tk.Scale(frame1, from_=0, to=total_frames-1,resolution=1, label="Navigation im Video",orient="horizontal", command=self.set_frame, length=300)
        #self.video_slider.pack()
        self.video_slider.grid(row=0,column=0,pady=5)
        self.video_slider.set(current_frame)

        self.video_slider.bind("<ButtonRelease-1>", self.updateValue)
        
        #Head09 = tk.Label(f1l, text="Framenummer")
        #Head09.pack(anchor='w')
        
        
        self.next_button = tk.Button(frame1, text=">", command=self.next_frame)
        #self.next_button.pack(side='right',anchor='se', expand=True)
        self.next_button.grid(row=0,column=2,pady=5)
        
        self.prev_button = tk.Button(frame1, text="<", command=self.prev_frame)
        #self.prev_button.pack(side='left',anchor='sw',expand=True)
        self.prev_button.grid(row=0,column=1,pady=5)

############################## FRAME 1a KO-SYSTEM UND POLAR PLOT UND OPTIONEN
        frame1a=tk.Frame(root,width=450,borderwidth=2,relief='sunken')
        frame1a.grid(row=1,column=0,padx=10,sticky='ew')
        frame1a.columnconfigure(0,weight=1)
        frame1a.columnconfigure(1,weight=2)
        frame1a.columnconfigure(2,weight=2)
        frame1a.columnconfigure(3,weight=2)
        frame1a.columnconfigure(4,weight=2)
        frame1a.columnconfigure(5,weight=1)

        self.start_button = tk.Button(frame1a, text="KO-System \n ausrichten", width=10, height=3, command=self.mark_start)
        self.start_button.grid(row=0,column=0,pady=5, rowspan=2)
        if platform.system() == "Darwin":   ### if its a Mac
            self.orig_color=self.start_button.cget("highlightbackground")
        else:
            self.orig_color=self.start_button.cget("background")

        self.kosys_box = tk.Checkbutton(frame1a,command=self.select_kosys)
        self.kosys_box["text"] = "KO-System"
        self.kosys_box.grid(row=0,column=1,sticky="w")
        self.kosys = tk.BooleanVar()
        self.kosys.set(True)
        self.kosys_box["variable"] = self.kosys
        
        self.scale_box = tk.Checkbutton(frame1a,command=self.select_scale)
        self.scale_box["text"] = "Skala"
        self.scale_box.grid(row=1,column=1,sticky="w")
        self.scale = tk.BooleanVar()
        self.scale.set(False)
        self.scale_box["variable"] = self.scale 


        self.polar_button = tk.Button(frame1a, text="Polar \n Plot", width=10, height=3, command=self.select_polarmethod)
        self.polar_button.grid(row=0,column=2,pady=5, rowspan=2)

        self.polarscale_deg = tk.BooleanVar()
        self.polaerscale_deg_button = tk.Checkbutton(frame1a,variable = self.polarscale_deg, offvalue=False, onvalue=True, text="Grad    Radian",command=self.setdegscale)
        self.polaerscale_deg_button.grid(row=0,column=3, sticky='w')
        self.polarscale_deg.set(True)

        self.polarscale_rad = tk.BooleanVar()
        self.polaerscale_rad_button = tk.Checkbutton(frame1a,variable = self.polarscale_rad, offvalue=False, onvalue=True, text="",command=self.setradscale)
        self.polaerscale_rad_button.grid(row=0,column=4, sticky='w')
        self.polarscale_rad.set(False)

        self.polarscale_length = tk.BooleanVar()
        self.polaerscale_length_button = tk.Checkbutton(frame1a,variable = self.polarscale_length, offvalue=False, onvalue=True, text="Skala in m/cm",command=self.setlengthscale)
        self.polaerscale_length_button.grid(row=1,column=3, sticky='w')
        self.polarscale_length.set(False)

        DISABLED="disabled"
        self.polaerscale_deg_button.config(state=DISABLED)
        self.polaerscale_rad_button.config(state=DISABLED)
        self.polaerscale_length_button.config(state=DISABLED)

############################## FRAME 1B UND 1C: MASSTAB FESTLEGEN UND REDRAW
        frame1BC=tk.Frame(root,width=450)
        frame1BC.grid(row=2,column=0,padx=0,pady=0,sticky='ew')
        
        frame1BC.columnconfigure(0,weight=1)
        frame1BC.columnconfigure(1,weight=1)
        
        frame1b=tk.Frame(frame1BC,width=205,borderwidth=2,relief='sunken')
        frame1b.grid(row=0,column=0,padx=10,pady=10,sticky='nsew')

        frame1c=tk.Frame(frame1BC,width=205,borderwidth=2,relief='sunken')
        frame1c.grid(row=0,column=1,padx=10,pady=10,sticky='nsew')

        self.length_button = tk.Button(frame1b, text="Maßstab \n festlegen", height=2, width=10, command=self.mark_length)
        self.length_button.grid(row=0,column=0,pady=5,rowspan=2, columnspan=2)

        self.cal_label = tk.Label(frame1b, text="Länge des Maßstabs= ")
        #self.cal.pack(side='left')
        self.cal_label.grid(row=2,column=0,sticky='e',columnspan=3)
        self.cal = tk.Entry(frame1b,width=5)
        self.cal.insert(0,'1.0')
       # self.cal.pack(side='left')
        self.cal.grid(row=2,column=3,sticky='w')

        self.insert_glmbesch = tk.Button(frame1b, text="a=const.", width=8, height=2, command=self.glm_besch)
        self.insert_glmbesch.grid(row=0,column=2,pady=5, rowspan=2,columnspan=2)


        frame1c.columnconfigure(0, weight=2)
        frame1c.columnconfigure(1, weight=1)
        frame1c.columnconfigure(2, weight=2)
        
        self.insert_scale_button = tk.Button(frame1c, text="KO-Sytem neu \n zeichnen", width=13, height=3, command=self.redraw)
        self.insert_scale_button.grid(row=0,column=0,rowspan=3,pady=10)

        self.remove_scale_button = tk.Button(frame1c, text="KO-System \n entfernen", width=8, height=2,command=self.remove_scale)
        self.remove_scale_button.grid(row=0,column=2,rowspan=3)


############## FRAME2
        frame2=tk.Frame(root,width=450,borderwidth=2,relief='sunken')
        frame2.grid(row=3,column=0,padx=10,sticky='ew')

        frame2.columnconfigure(0,weight=2)
        frame2.columnconfigure(1,weight=1)
        frame2.columnconfigure(2,weight=1)
        frame2.columnconfigure(3,weight=1)
        frame2.columnconfigure(4,weight=1)
        frame2.columnconfigure(5,weight=1)
        frame2.columnconfigure(6,weight=1)


        self.kleinster_wert_label = tk.Label(frame2, text="x_min,x_max:")
        self.kleinster_wert_label.grid(row=0,column=0)
        self.kleinster_wert_entry = tk.Entry(frame2,width=8)
        self.kleinster_wert_entry.insert(0, '0.0 , 1.0')
        self.kleinster_wert_entry.grid(row=0,column=1,sticky='w')

        self.ybounds_label = tk.Label(frame2,text="y_min,y_max:")
        self.ybounds_label.grid(row=1,column=0)
        self.ybounds_entry = tk.Entry(frame2,width=8)
        self.ybounds_entry.insert(0,'0.0 , 1.0')
        self.ybounds_entry.grid(row=1,column=1,sticky='w')

        self.lower_marker = tk.Label(frame2, text="Anheftpunkt:")
        self.lower_marker.grid(row=2,column=0,pady=5)
        self.lower_marker = tk.Entry(frame2, width=8)
        self.lower_marker.insert(0, '0.0 , 0.0')
        self.lower_marker.grid(row=2,column=1,sticky='w')
        self.anheft=self.lower_marker.get()


        self.zerolabel = tk.BooleanVar()
        self.zerolabel_button = tk.Checkbutton(frame2,variable = self.zerolabel, offvalue=False, onvalue=True, text="Label bei x=0",command=self.redraw)
        self.zerolabel_button.grid(row=0,column=2, sticky='w')
        self.zerolabel.set(False)

        self.show_xlabels_box = tk.Checkbutton(frame2)
        self.show_xlabels_box["text"] = "x-Zahlen"
        self.show_xlabels_box.grid(row=1,column=2,sticky='w')
        self.show_xlabels = tk.BooleanVar()
        self.show_xlabels.set(True)
        self.show_xlabels_box["variable"] = self.show_xlabels        

        self.show_ylabels_box = tk.Checkbutton(frame2)
        self.show_ylabels_box["text"] = "y-Zahlen"
        self.show_ylabels_box.grid(row=2,column=2, sticky='w')
        self.show_ylabels = tk.BooleanVar()
        self.show_ylabels.set(True)
        self.show_ylabels_box["variable"] = self.show_ylabels   

        
        self.bg = tk.BooleanVar()
        self.bg_button = tk.Checkbutton(frame2,variable = self.bg, offvalue=False, onvalue=True, text="weißer Hintergrund",command=self.white_bg)
        self.bg_button.grid(row=2,column=3,sticky="w")
        self.bg.set(False)

        self.invert_y = tk.Checkbutton(frame2,command=self.inverty)
        self.invert_y["text"] = "y-Achse umkehren"
        self.invert_y.grid(row=1,column=3,sticky="w")
        self.invertvar_y = tk.BooleanVar()
        self.invertvar_y.set(False)
        self.invert_y["variable"] = self.invertvar_y
        
        self.lastlabel = tk.Checkbutton(frame2)
        self.lastlabel["text"] = "Label bei letzem Wert"
        self.lastlabel.grid(row=0,column=3,sticky="w")
        self.lastlabelvar = tk.BooleanVar()
        self.lastlabelvar.set(True)
        self.lastlabel["variable"] = self.lastlabelvar
        self.lastlabel["command"]=self.redraw
        
######################### FRAME3
        frame3=tk.Frame(root,width=450,borderwidth=2,relief='sunken')
        frame3.grid(row=4,column=0,padx=10,pady=10,sticky='ew')

        frame3.columnconfigure(0,weight=2)
        frame3.columnconfigure(1,weight=1)
        frame3.columnconfigure(2,weight=2)
        frame3.columnconfigure(3,weight=1)


        self.labels_label = tk.Label(frame3, text="Abstand der Labels:")
        self.labels_label.grid(row=0,column=0)
        self.labels_label_entry = tk.Entry(frame3,width=5)
        self.labels_label_entry.insert(0, '0.1')
        self.labels_label_entry.grid(row=0,column=1)
        
        self.longticks = tk.Label(frame3, text="Abstand große Striche:")
        self.longticks.grid(row=0,column=2)
        self.longticks_entry = tk.Entry(frame3,width=5)
        self.longticks_entry.insert(0, '0.05')
        self.longticks_entry.grid(row=0,column=3)
        
        self.lsize_titel = tk.Label(frame3, text="Textgröße der Labels:")
        self.lsize_titel.grid(row=1,column=0)
        self.lsize = tk.Entry(frame3,width=5)
        self.lsize.insert(0, '12')
        self.lsize.grid(row=1,column=1)

        self.shortticks = tk.Label(frame3, text="Abstand kleine Striche:")
        self.shortticks.grid(row=1,column=2)
        self.shortticks_entry = tk.Entry(frame3,width=5)
        self.shortticks_entry.insert(0, '0.01')
        self.shortticks_entry.grid(row=1,column=3)

##################### FRAME4
        frame4=tk.Frame(root,width=450,borderwidth=2,relief='sunken')
        frame4.grid(row=5,column=0,padx=10,sticky='ew')

        frame4.columnconfigure(0,weight=1)
        frame4.columnconfigure(1,weight=1)
        frame4.columnconfigure(2,weight=1)
        frame4.columnconfigure(3,weight=1)
        frame4.columnconfigure(4,weight=1)
        frame4.columnconfigure(5,weight=1)
        frame4.columnconfigure(6,weight=1)
        frame4.columnconfigure(7,weight=1)
        frame4.columnconfigure(8,weight=1)
        frame4.columnconfigure(9,weight=1)
        frame4.columnconfigure(10,weight=1)
        frame4.columnconfigure(11,weight=1)




        self.grayscale_slider = tk.Scale(frame4, label="Graustufen Gitter", showvalue=0, from_=0.1, to=1.0, resolution=.01, orient="horizontal", length=110)
        self.grayscale_slider.grid(row=0,column=0,columnspan=2, sticky='')
        self.grayscale_slider.set(1.0)

        self.grayscale_slider2 = tk.Scale(frame4, showvalue=0, from_=0.01, to=1.0, resolution=.01, orient="horizontal", length=110)
        self.grayscale_slider2.grid(row=1,column=0,columnspan=2, sticky='')
        self.grayscale_slider2.set(1.0)

        self.lw_slider = tk.Scale(frame4, label="Liniendicke Gitter", showvalue=0, from_=0.1, to=5.0, resolution=.1, orient="horizontal", length=110)
        self.lw_slider.grid(row=0,column=3,columnspan=2, sticky='')
        self.lw_slider.set(1.5)
        
        self.gw_slider = tk.Scale(frame4, showvalue=0, from_=0.1, to=5.0, resolution=.1, orient="horizontal", length=110)
        self.gw_slider.grid(row=1,column=3,columnspan=2, sticky='')
        self.gw_slider.set(1.0)

        
        
        self.ticklength = tk.Scale(frame4, label="Teilstrichlänge", showvalue=0, from_=0, to=25, orient="horizontal", length=110, resolution=1)
        self.ticklength.grid(row=0,column=6,columnspan=2, sticky='')
        self.ticklength.set(10)

        self.alpha_slider = tk.Scale(frame4, label="Transparenz", showvalue=0, from_=0, to=1., orient="horizontal", length=110, resolution=.01,command=self.change_alpha)
        self.alpha_slider.grid(row=0,column=9,columnspan=2, sticky='')
        self.alpha_slider.set(0)


#################### Frame 5 Timestamp

        frame5=tk.Frame(root,width=450,borderwidth=2,relief='sunken')
        frame5.grid(row=6,column=0,padx=10,sticky='ew')

######### Alte Labels        
        #self.t0_frame_label = tk.Label(frame5, text="t=0 bei Frame: 0000")
        #self.t0_frame_label.grid(row=0,column=0, columnspan=1, sticky='w')
        
        self.time_label = tk.Label(frame5, text="t ab Frame 1: 00:00.000")
        #self.time_label.grid(row=0,column=1,columnspan=2,sticky='ew')
        
        self.time_since_t0_label = tk.Label(frame5, text="t ab t=0: 00.00.000")
        #self.time_since_t0_label.grid(row=0,column=3, columnspan=2,sticky='w')

######## Neue Labels
        self.timeshift_label = tk. Label(frame5,text="Zeit verschieben:")
        self.timeshift_label.grid(row=0,column=0,columnspan=2, sticky='w')
        
        self.timescale_label = tk. Label(frame5,text="Zeit skalieren:")
        self.timescale_label.grid(row=1,column=0,columnspan=2, sticky='w')


        self.t0_label = tk.Label(frame5, text="t0=")
        self.t0_label.grid(row=0,column=2, columnspan=1, sticky='w')
        
        self.t0_entry = tk.Entry(frame5,width=5)
        self.t0_entry.grid(row=0,column=3)
        self.t0_entry.insert(0,'0')
        
        self.t0_frame_label = tk.Label(frame5,text="bei Frame 000 ")
        self.t0_frame_label.grid(row=0,column=4, sticky='w')
        
        self.t0_button = tk.Button(frame5, text="setzen", width=10, height=1, command=self.set_t0)
        self.t0_button.grid(row=0,column=5,sticky='w')
        
        self.t1_Label = tk.Label(frame5, text="t1=")
        self.t1_Label.grid(row=1,column=2, sticky='w')
                
        self.t1_entry = tk.Entry(frame5,width=5)
        self.t1_entry.grid(row=1,column=3)
        self.t1_entry.insert(0,'0')
        
        self.t1_frame_label = tk.Label(frame5,text="bei Frame 000 ")
        self.t1_frame_label.grid(row=1,column=4, sticky='w')
        
        self.t1_button = tk.Button(frame5, text="setzen", width=10, height=1, command=self.set_t1)
        self.t1_button.grid(row=1,column=5,rowspan=1)
        
                
        


        self.add_time_button = tk.Button(frame5, text="Zeitstempel \n einfügen", width=10, height=3, command=self.add_timestamp)
        self.add_time_button.grid(row=3,column=0,rowspan=3,columnspan=2)


        self.Timestamp_size=tk.Scale(frame5,from_=0, to=2, resolution=.1,  orient=tk.HORIZONTAL, length=185, label="Größe des Zeitstempels",command=self.changeTSsize)
        self.Timestamp_size.grid(row=3,column=2,columnspan=3,rowspan=3)
        if timestamp_size is not None:
            self.Timestamp_size.set(timestamp_size)
        else:
            self.Timestamp_size.set(1.0)
        
        #self.Timestamp_angle=tk.Scale(frame5,from_=-90, to=90,  orient=tk.HORIZONTAL, length=185, label="Rotationswinkel Zeitstempel",command=self.changeTSsize)
        self.Timestamp_angle=tk.Scale(frame5,from_=-90, to=90, resolution=2,  orient=tk.HORIZONTAL, length=185, label="Rotationswinkel Zeitstempel",command=self.changeTSsize)

        
        self.Timestamp_angle.grid(row=6,column=2,columnspan=3,rowspan=3)
        if timestamp_angle is not None:
            self.Timestamp_angle.set(timestamp_angle)
        else:
            self.Timestamp_angle.set(0.0)

        self.native_framerate_lab = tk.Label(frame5, text="fps original:")
        self.native_framerate_lab.grid(row=3,column=5,sticky='w',columnspan=2)

        self.native_framerate_label = tk.Label(frame5,text=" ")
        self.native_framerate_label.grid(row=3,column=7)

        self.framerate_lab = tk.Label(frame5, text="fps aktuell:")
        self.framerate_lab.grid(row=4,column=5,sticky='w',columnspan=2)

        self.framerate_label = tk.Label(frame5,text=" ")
        self.framerate_label.grid(row=4,column=7)

        self.fps_new = tk.Label(frame5, text="Setze fps:")
        self.fps_new.grid(row=5,column=5, sticky='w',columnspan=2)


        self.fps_new_entry = tk.Entry(frame5,width=5)
        self.fps_new_entry.grid(row=5,column=7)

        self.remove_timestamp_button=tk.Button(frame5,text="...entfernen",width=10,height=2,command=self.remove_timestamp)
        self.remove_timestamp_button.grid(row=6,column=0,columnspan=2)


        #self.timeshift_box = tk.Checkbutton(frame5,command=self.select_timeshift)
        #self.timeshift_box["text"] = "Zeit-shift / scale"
        #self.timeshift_box.grid(row=6,column=4,sticky="w",columnspan=2)
        #self.timeshift = tk.BooleanVar()
        #self.timeshift.set(True)
        #self.timeshift_box["variable"] = self.timeshift
        
        #self.timescale_box = tk.Checkbutton(frame5,command=self.select_timescale)
        #self.timescale_box["text"] = "Zeit shiften / sklaieren"
        #self.timescale_box.grid(row=6,column=6,sticky="w")
        #self.timescale = tk.BooleanVar()
        #self.timescale.set(False)
        #self.timescale_box["variable"] = self.timescale
        

        


################### Frame 6 Fertig
        frame6=tk.Frame(root,width=450,borderwidth=2,relief='sunken')
        frame6.grid(row=7,column=0,padx=10,sticky='ew')
        
        frame6.columnconfigure(0,weight=1)
        frame6.columnconfigure(1,weight=1)
        frame6.columnconfigure(2,weight=1)
        frame6.columnconfigure(3,weight=1)
        frame6.columnconfigure(4,weight=1)
        frame6.columnconfigure(5,weight=1)
        frame6.columnconfigure(6,weight=1)
        frame6.columnconfigure(7,weight=1)
        frame6.columnconfigure(8,weight=1)
        frame6.columnconfigure(9,weight=1)
        
        self.frame_min_label=tk.Label(frame6,text="Erstes Bild bei Nr.:")
        self.frame_min_label.grid(row=0,column=0,columnspan=3,sticky='w')

        self.frame_max_label=tk.Label(frame6,text="Letzes Bild bei Nr.:")
        self.frame_max_label.grid(row=0,column=4,columnspan=3,sticky='w')
        

        self.crop_button=tk.Button(frame6,text='Crop',width=5,height=3,command=self.Crop)
        self.crop_button.grid(row=1,column=0,rowspan=3,sticky='ew')

        self.frame0_button=tk.Button(frame6,text='Bild als \n 1. Bild',width=5,height=3,command=self.set_frame_min)
        self.frame0_button.grid(row=1,column=1,rowspan=3, sticky='ew')

        self.last_frame_button=tk.Button(frame6,text='Bild als \n letztes \n Bild',width=5,height=3,command=self.set_frame_max)
        self.last_frame_button.grid(row=1,column=2,rowspan=3, sticky='ew')

        self.step_label=tk.Label(frame6,text="Schrittweite beim Export:")
        self.step_label.grid(row=4,column=0,columnspan=2,sticky='w')
        self.step_entry=tk.Entry(frame6,width=5)
        self.step_entry.grid(row=4,column=2,sticky='w')
        self.step_entry.insert(0,"1")
        
        self.save_button=tk.Button(frame6,text='Save Video / Image',width=15,height=4,command=self.save_video_with_scale)
        self.save_button.grid(row=1,column=4,columnspan=3,rowspan=4,sticky='ns')

        self.save_scale_button=tk.Button(frame6,text='Export \n KO-Sys as \n png',width=8,height=3,command=self.save_scale)
        self.save_scale_button.grid(row=1,column=7,columnspan=2, rowspan=3,sticky='ns')

        DISABLED="disabled"
        if imageflag:
            self.insert_glmbesch.config(state=DISABLED)
            self.add_time_button.config(state=DISABLED)
            self.t0_button.config(state=DISABLED)
            self.remove_timestamp_button.config(state=DISABLED)
            self.frame0_button.config(state=DISABLED)
            self.last_frame_button.config(state=DISABLED)
            self.Timestamp_size.config(state=DISABLED)


############################ MENUE ENDE ######################################



        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    def entries_loose_focus(self,event):
        if not isinstance(event.widget, tk.Entry):
            self.frame_max_label.focus_set()
        
    
    def move_scale_left(self,event):
        global start_point,start_x,start_y, end_x, end_y
        if not scale_added:
            return
        else:
            if start_x-1>=0:
                start_x=start_x-1
                end_x=end_x-1
                x,y=self.end_point
                x=x-1
                self.end_point=(x,y)
                self.start_point=(start_x,start_y)
                start_point=self.start_point
                self.show_first_frame()
                
    def move_scale_right(self,event):
        global start_point,start_x,start_y, end_x, end_y
        if not scale_added:
            return
        else:
            if start_x+1<=self.first_frame.shape[1]:
                start_x=start_x+1
                end_x=end_x+1
                x,y=self.end_point
                x=x+1
                self.end_point=(x,y)
                self.start_point=(start_x,start_y)
                start_point=self.start_point
                self.show_first_frame()
    
    def move_scale_down(self,event):
        global start_point,start_x,start_y, end_x, end_y
        if not scale_added:
            return
        else:
            if start_y+1<=self.first_frame.shape[0]:
                start_y=start_y+1
                end_y=end_y+1
                x,y=self.end_point
                y=y+1
                self.end_point=(x,y)
                self.start_point=(start_x,start_y)
                start_point=self.start_point
                self.show_first_frame()
    
    def move_scale_up(self,event):
        global start_point,start_x,start_y, end_x, end_y
        if not scale_added:
            return
        else:
             if start_y-1>=0:
                start_y=start_y-1
                end_y=end_y-1
                x,y=self.end_point
                y=y-1
                self.end_point=(x,y)
                self.start_point=(start_x,start_y)
                start_point=self.start_point
                self.show_first_frame()
                
    def rotate_left(self,event):
        global start_point,start_x,start_y, end_x, end_y, scale_image, y_shift, x_shift, scale_markup
        xs,ys=self.start_point
        xe,ye=self.end_point_bak
        
        
        angle=0.25
        self.keyrot_angle=self.keyrot_angle+angle
        #print("START POINT",xs,ys)
        #print("END PONT",self.end_point,xe,ye)
        x,y=xe-xs,ye-ys
        #print("x,y",x,y)
        #radians=np.pi/180.*10.

        radians=np.radians(self.keyrot_angle)
        c, s = np.cos(radians), np.sin(radians)
        j = np.matrix([[c, s], [-s, c]])
        m = np.dot(j, [x, y])
        #print(m)
        dx,dy=int(round(m[0,0],0)), int(round(m[0,1],0))
        #print("dx,dy",dx,dy)
        self.end_point=(xs+dx,ys+dy)
        end_x,end_y=self.end_point
        #print("END PONT2",self.end_point,xe,ye)
        if self.rot_angle_bak is not None:
            scale_image = self.scale_image_not_rot.copy()
            scale_image = self.rotate_image(self.rot_angle_bak-angle)
            
        
        self.show_first_frame()
        
    def rotate_right(self,event):
        global start_point,start_x,start_y, end_x, end_y, scale_image, y_shift, x_shift, scale_markup
        xs,ys=self.start_point
        xe,ye=self.end_point_bak
        
        angle=0.25
        self.keyrot_angle=self.keyrot_angle-angle
        #print("START POINT",xs,ys)
        #print("END PONT",self.end_point,xe,ye)
        x,y=xe-xs,ye-ys
        #print("x,y",x,y)
        #radians=np.pi/180.*10.

        radians=np.radians(self.keyrot_angle)
        c, s = np.cos(radians), np.sin(radians)
        j = np.matrix([[c, s], [-s, c]])
        m = np.dot(j, [x, y])
        #print(m)
        dx,dy=int(round(m[0,0],0)), int(round(m[0,1],0))
        #print("dx,dy",dx,dy)
        self.end_point=(xs+dx,ys+dy)
        end_x,end_y=self.end_point
        #print("END PONT2",self.end_point,xe,ye)
        if self.rot_angle_bak is not None:
            scale_image = self.scale_image_not_rot.copy()
            scale_image = self.rotate_image(self.rot_angle_bak+angle)
            
        
        self.show_first_frame()        
                
            
        
        
    
    def setdegscale(self):
        self.polarscale_length.set(False)
        self.polarscale_rad.set(False)
        self.kleinster_wert_label.config(text=f"r_min,r_max:")
        self.ybounds_label.config(text=f"theta_min, theta_max:")
        self.ybounds_entry.delete(0,tk.END)
        self.ybounds_entry.insert(0,'0.0 , 360.0')
        self.labels_label_entry.delete(0,tk.END)
        self.labels_label_entry.insert(0, '45')
        self.longticks_entry.delete(0,tk.END)
        self.longticks_entry.insert(0,'45')
        self.shortticks_entry.delete(0,tk.END)
        self.shortticks_entry.insert(0,'5')


    def setradscale(self):
        self.polarscale_length.set(False)
        self.polarscale_deg.set(False)
        self.kleinster_wert_label.config(text=f"r_min,r_max:")
        self.ybounds_label.config(text=f"theta_min, theta_max:")
        self.ybounds_entry.delete(0,tk.END)
        self.ybounds_entry.insert(0,'0.0 , 360.0')
        self.labels_label_entry.delete(0,tk.END)
        self.labels_label_entry.insert(0, '45')
        self.longticks_entry.delete(0,tk.END)
        self.longticks_entry.insert(0,'45')
        self.shortticks_entry.delete(0,tk.END)
        self.shortticks_entry.insert(0,'5')

    def setlengthscale(self):
        self.polarscale_rad.set(False)
        self.polarscale_deg.set(False)

        rax = self.kleinster_wert_entry.get()
        #xmax = float(self.groesster_wert_entry.get())
        rmin,rmax=map(float,rax.split(','))

        self.kleinster_wert_label.config(text=f"r_min,r_max:")
        self.ybounds_label.config(text=f"s_min, s_max:")
        self.ybounds_entry.delete(0,tk.END)
        smax=np.round(2*np.pi*rmax,2)
        self.ybounds_entry.insert(0,f'0.0 , {smax}')
        self.labels_label_entry.delete(0,tk.END)
        self.labels_label_entry.insert(0, f'{np.round(smax/8.,2)}')
        self.longticks_entry.delete(0,tk.END)
        self.longticks_entry.insert(0,f'{np.round(smax/8.,2)}')
        self.shortticks_entry.delete(0,tk.END)
        self.shortticks_entry.insert(0,f'{np.round(smax/80.,2)}')
        
        
    

    
    def updateValue(self,Value):
        global current_frame,cached_frames
        current_frame=self.video_slider.get()
        self.first_frame=cached_frames[current_frame].copy()
        self.show_first_frame()
        cv2.waitKey(1)

    def remove_timestamp(self):
        global timestamp_added
        timestamp_added=False
        self.show_first_frame()
    
    def remove_scale(self):
        global scale_added
        scale_added=False
        self.show_first_frame()
    
    def update_control_panel(self,total_frames,fps):
        global cached_frames, current_frame,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added, timestamp_position    
        
        #if fps_new_entry.get() !="":
        #    fps=float(fps_new_entry.get())
        #    fps=float(fps_new_entry.get())
        #else:
        #    self.fps=self.cap.get(cv2.CAP_PROP_FPS)
        #    fps=round(self.fps,2)
        natfps=round(fps_native,2)
        fps=round(fps,2)
        Hunderter=int(current_frame/100)
        Zehner=int((current_frame-Hunderter*100)/10)
        Einer=int((current_frame-Hunderter*100-Zehner*10))
        cf_str=f"{str(Hunderter)}{str(Zehner)}{str(Einer)}"
        
        self.t1_frame_label.config(text=f"bei Frame {cf_str}")
        self.framerate_label.config(text=f"{fps}")
        self.native_framerate_label.config(text=f"{natfps}")

        time_diff = current_frame/fps
        ten_minutes=int(time_diff/600.)
        time_diff=time_diff-float(ten_minutes)*600.
        minutes=int(time_diff/60.)
        time_diff=time_diff-float(minutes)*60.
        ten_seconds=int(time_diff/10.)
        time_diff=time_diff-float(ten_seconds)*10.
        seconds=int(time_diff)
        time_diff=time_diff-float(seconds)*1.0
        zehntel=int(time_diff/0.1)
        time_diff=time_diff-float(zehntel)*0.1
        hstel=int(time_diff/.01)
        time_diff=time_diff-float(hstel)*0.01
        tstel=int(time_diff/.001)
        timestamp_text = f"Time:  {str(ten_minutes)}{str(minutes)}:{str(ten_seconds)}{str(seconds)}.{str(zehntel)}{str(hstel)}{str(tstel)}"
        timetext=f"{str(ten_minutes)}{str(minutes)}:{str(ten_seconds)}{str(seconds)}.{str(zehntel)}{str(hstel)}{str(tstel)}"
 
        self.time_label.config(
            text=f"t ab Bild 0:    {timetext}")

        #self.t0_frame_label.config(text=f"t={self.t0_rest} bei Bild Nr.: {t0_frame}")
        self.frame_min_label.config(text=f"Erstes Bild bei Nr.: {self.frame_min}")
        self.frame_max_label.config(text=f"Letztes Bild bei Nr.: {self.frame_max}")

        #time_diff=(current_frame - t0_frame) / fps+self.t0_rest
        time_diff=round(current_frame/fps-self.t0,3)
        time_diff_str=str(time_diff)
        if time_diff<0:
            negativ=True
            time_diff=-1.*time_diff
        else:
            negativ=False
        ten_minutes=int(time_diff/600.)
        time_diff=time_diff-float(ten_minutes)*600.
        minutes=int(time_diff/60.)
        time_diff=time_diff-float(minutes)*60.
        ten_seconds=int(time_diff/10.)
        time_diff=time_diff-float(ten_seconds)*10.
        seconds=int(time_diff)
        time_diff=time_diff-float(seconds)*1.0
        zehntel=int(time_diff/0.1)
        time_diff=time_diff-float(zehntel)*0.1
        hstel=int(time_diff/.01)
        time_diff=time_diff-float(hstel)*0.01
        tstel=int(time_diff/.001)
        if not negativ:
            timestamp_text = f"Time:  {str(ten_minutes)}{str(minutes)}:{str(ten_seconds)}{str(seconds)}.{str(zehntel)}{str(hstel)}{str(tstel)}"
            frame_time=f"{str(ten_minutes)}{str(minutes)}:{str(ten_seconds)}{str(seconds)}.{str(zehntel)}{str(hstel)}{str(tstel)}"

        else:
            timestamp_text = f"Time: -{str(ten_minutes)}{str(minutes)}:{str(ten_seconds)}{str(seconds)}.{str(zehntel)}{str(hstel)}{str(tstel)}"
            frame_time=f"-{str(ten_minutes)}{str(minutes)}:{str(ten_seconds)}{str(seconds)}.{str(zehntel)}{str(hstel)}{str(tstel)}"

        
        self.time_since_t0_label.config(
            text=f"t ab    t=0:    {frame_time}")
        self.t1_entry.delete(0,tk.END)
        self.t1_entry.insert(0,time_diff_str)
        




    def set_t0(self):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added    
        
        if 1==1:
            #if self.fps_new_entry.get() != "":
            #    self.fps=float(self.fps_new_entry.get())
            #else:
            #    self.fps=self.cap.get(cv2.CAP_PROP_FPS)
            t0=float(self.t0_entry.get())  
            t0_frame = current_frame
            t0_time = current_frame / fps-t0
            self.t0=t0_time
            self.t0_frame_label.config(text=f"bei Frame: {current_frame}")
            self.show_first_frame()
            
    def set_t1(self):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added    
        
        global fps
        t1=float(self.t1_entry.get())
        cf=current_frame
        ct=cf/fps
        zf=t0_frame
        t_zeroframe=zf/fps-self.t0
        #if self.timescale.get():
        fps_old=fps
        #print("t0-time BEFORE0",t0_time,"zeroframe",zf)
        fps=round(float(cf-zf)/(t1-t_zeroframe),2)
        #self.fps=fps
        #self.fps_aktuell=fps
        #t0_time=t0_time*fps_old/fps
        #print("t0-time BEFORE",t0_time,"zeroframe",zf)
        t0_time=zf/fps-(zf/fps_old-t0_time)
        #print("t-Ziel",t1,"t0",t0_time)
        self.t0=t0_time
        self.fps_new_entry.delete(0,tk.END)
        #self.fps_new_entry.insert(0,str(fps))
        #self.t1_entry.delete(0,tk.END)
        self.update_control_panel(total_frames,fps)
        self.show_first_frame()

            
    


    def changeTSsize(self,value):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added, timestamp_angle
        if cached_frames is not None:
            self.show_first_frame()

    def reset_all_buttons(self):
        if platform.system() == "Darwin":   ### if its a Mac
            self.add_time_button.config(relief='raised',highlightbackground=self.orig_color)
            self.start_button.config(relief='raised',highlightbackground=self.orig_color)
            self.length_button.config(relief='raised',highlightbackground=self.orig_color)
            self.insert_glmbesch.config(relief='raised',highlightbackground=self.orig_color)
            self.crop_button.config(relief='raised',highlightbackground=self.orig_color)
        else:
            self.add_time_button.config(relief='raised',bg=self.orig_color)
            self.start_button.config(relief='raised',bg=self.orig_color)
            self.length_button.config(relief='raised',bg=self.orig_color)
            self.insert_glmbesch.config(relief='raised',bg=self.orig_color)
            self.crop_button.config(relief='raised',bg=self.orig_color)  

    def glm_besch(self):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
        if cached_frames is not None:
            self.reset_all_buttons()
            if platform.system() == "Darwin":   ### if its a Mac
                self.insert_glmbesch.config(relief='sunken',highlightbackground='violet')
            else:
                self.insert_glmbesch.config(relief='sunken',bg='violet')
            cv2.destroyWindow("Video")
            yn=messagebox.askquestion('a = const','Soll ein Wert für a festgelegt werden?')
            if yn=='yes':
                astr = simpledialog.askstring(title="a = const.", prompt="Bitte geben Sie die Beschleunigung ein!")
                if astr=="":
                    return
                self.beschleunigung=float(astr)
                self.choose_a=True
            else:
                self.choose_a=False
            
            yn=messagebox.askquestion('Achse der Beschleunigung','Achse der Beschleunigung horizontal im Bild?')
            if yn=='yes':
                self.a_along_x=True
            else:
                self.a_along_x=False
                
            yn=messagebox.askquestion('Achse der Beschleunigung','Achse der Beschleunigung vertikal im Bild?')
            if yn=='yes':
                self.a_along_y=True
            else:
                self.a_along_y=False
            
            #self.t0_frame_buffer=t0_frame.copy()
            
            
            #self.set_frame(0)
            self.show_first_frame()
            #x,y,w,h=cv2.getWindowImageRect("Video Editor")
            x=self.root.winfo_x()
            y=self.root.winfo_y()
            w=self.root.winfo_width()
            cv2.moveWindow("Video",x+w+30,y)
            cv2.waitKey(1)
            self.mark_apoints()


    def mark_apoints(self):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
        cv2.destroyWindow("Video")
        messagebox.showinfo("ZWEI POSITIONEN DURCH KLCK MARKIEREN!","Bitte wählen Sie die erste Position aus!")
        self.set_frame(current_frame)
        self.show_first_frame()
        #x,y,w,h=cv2.getWindowImageRect("Video Editor")
        x=self.root.winfo_x()
        y=self.root.winfo_y()
        w=self.root.winfo_width()
        cv2.moveWindow("Video",x+w+30,y)

        #cv2.setWindowProperty("Video", cv2.WND_PROP_TOPMOST, 1)
        cv2.setMouseCallback("Video", self.marka)
        cv2.setWindowTitle("Video","POSITION MARKIEREN (KLICK INS BILD)")
        

    
    def marka(self, event, x, y, flags, param):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.marka1:
                self.pos1 = (x, y)
                self.glmbesch_t1=(current_frame) / fps
                self.marka1=False
                cv2.destroyWindow("Video")
                messagebox.showinfo("ZWEI POSITIONEN DURCH KLCK MARKIEREN!","Bitte wählen Sie die zweite Position aus!")
                self.show_first_frame()
                #x,y,w,h=cv2.getWindowImageRect("Video Editor")
                x=self.root.winfo_x()
                y=self.root.winfo_y()
                w=self.root.winfo_width()
                cv2.moveWindow("Video",x+w+30,y)
                #cv2.setWindowProperty("Video", cv2.WND_PROP_TOPMOST, 1)
                cv2.setMouseCallback("Video", self.marka)
                cv2.setWindowTitle("Video","POSITION MARKIEREN (KLICK INS BILD)")
                self.marka2=True
            else:
                self.pos2 = (x, y)
                self.glmbesch_t2=(current_frame) / fps
                self.marka2=False
                self.marka1=True
                self.glm_besch2()


    def glm_besch2(self):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
        global start_x,start_y,scale_image,lnorm
        x=start_x
        y=start_y
        x1,y1=self.pos1
        x2,y2=self.pos2
        if self.a_along_x:
            s1=np.sqrt((x1-x)**2.)
            s2=np.sqrt((x2-x)**2.)
        elif self.a_along_y:
            s1=np.sqrt((y1-y)**2.)
            s2=np.sqrt((y2-y)**2.)
        else:
            s1=np.sqrt((x1-x)**2.+(y1-y)**2.)
            s2=np.sqrt((x2-x)**2.+(y2-y)**2.)
            
        t1=self.glmbesch_t1
        t2=self.glmbesch_t2
        t0=(np.sqrt(s1)*t2-np.sqrt(s2)*t1)/(np.sqrt(s1)-np.sqrt(s2))
        t0_frame=int(round(fps*t0,0))
        self.t0_rest=t0-t0_frame/fps
        self.t0=t0
        t0_time=t0
        print("t0",self.t0)
        
        cv2.destroyWindow("Video")
        messagebox.showinfo("a=const.ASSISTENT:",f"Der Anfang der Bewegung (t={self.t0_rest}) wurde auf Frame {t0_frame} gesetzt.")
        self.show_first_frame()
        #x,y,w,h=cv2.getWindowImageRect("Video Editor")
        x=self.root.winfo_x()
        y=self.root.winfo_y()
        w=self.root.winfo_width()
        cv2.moveWindow("Video",x+w+30,y)
        print("T0",t0,t0*fps,t0_frame)
        if self.choose_a:
            print("UNITS PER PIXEL BEFORE",self.units_per_px,self.beschleunigung, t1,s1)
            self.units_per_px=0.5*self.beschleunigung*(t1-t0)**2./s1
            print("UNITS PER PIXEL AFTER",self.units_per_px)
            
            cal=round(lnorm*self.units_per_px,2)
            self.cal.delete(0,tk.END)
            self.cal.insert(0,f"{cal}")
            self.units_per_px=None
            
            self.redraw()
            
            
            #self.insert_scale()
            #self.scaling_factor=1./self.units_per_px*self.markerdist_in_m/self.markerdist_in_pixels_as_is
            #scale_image=self.scale_image_not_rot.copy()
            #cv2.imshow("Test1",scale_image)
            #scale_image = self.rotate_image(self.rot_angle_bak)
            #cv2.imshow("Test",scale_image)
            
            #self.show_first_frame()

        


    def draw_timestamp(self, frame):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added, timestamp_size, timestamp_position, timestamp_angle
        #time_diff=(current_frame - t0_frame) / fps + self.t0_rest
        time_diff=current_frame/fps-self.t0
        if time_diff<0: 
            negativ=True
            time_diff=-1.*time_diff
        else:
            negativ=False
        ten_minutes=int(time_diff/600.)
        time_diff=time_diff-float(ten_minutes)*600.
        minutes=int(time_diff/60.)
        time_diff=time_diff-float(minutes)*60.
        ten_seconds=int(time_diff/10.)
        time_diff=time_diff-float(ten_seconds)*10.
        seconds=int(time_diff)
        time_diff=time_diff-float(seconds)*1.0
        zehntel=int(time_diff/0.1)
        time_diff=time_diff-float(zehntel)*0.1
        hstel=int(time_diff/.01)
        time_diff=time_diff-float(hstel)*0.01
        tstel=int(time_diff/.001)
        if not negativ:
            timestamp_text = f"Time:  {str(ten_minutes)}{str(minutes)}:{str(ten_seconds)}{str(seconds)}.{str(zehntel)}{str(hstel)}{str(tstel)}"
        else:
            timestamp_text = f"Time: -{str(ten_minutes)}{str(minutes)}:{str(ten_seconds)}{str(seconds)}.{str(zehntel)}{str(hstel)}{str(tstel)}"
        # Berechne die Breite und Höhe des Zeitstempelhintergrunds
        text_scale=self.Timestamp_size.get()
        timestamp_size=self.Timestamp_size.get()
        timestamp_angle=self.Timestamp_angle.get()
        #text_scale=1.0
        text_width, text_height = cv2.getTextSize(timestamp_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 2)[0]
        background_width = int(text_width * 1.1)
        background_height = int(text_height * 2.5)

        # Zeichne ein Rechteck mit abgerundeten Ecken
        rot_ts_frame = np.zeros_like(frame)
        rot_ts_frame1 = np.zeros_like(frame)
        ts_frame=np.zeros((frame.shape[0]+background_height,frame.shape[1]+background_width,frame.shape[2]), dtype=np.uint8)
        ts_frame1=np.zeros((frame.shape[0]+background_height,frame.shape[1]+background_width,frame.shape[2]), dtype=np.uint8)
        ts_rotation_matrix= cv2.getRotationMatrix2D(timestamp_position,timestamp_angle,1)
        
        #cv2.rectangle(ts_frame, timestamp_position,
         #             (timestamp_position[0] + background_width, timestamp_position[1] + background_height),
          #            (255, 255, 255), -1, cv2.LINE_AA, shift=0)
        cv2.rectangle(ts_frame, timestamp_position,
                      (timestamp_position[0] + background_width, timestamp_position[1] + background_height),
                      (255, 255, 255), -1)
        cv2.rectangle(ts_frame1, (timestamp_position[0]+2,timestamp_position[1]+2),
                      (timestamp_position[0] + background_width-2, timestamp_position[1] + background_height-2),
                      (255, 255, 255), -1)
        cv2.putText(ts_frame, timestamp_text, (timestamp_position[0] + int(10*text_scale), timestamp_position[1] + int(text_scale*40)),
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (1, 1, 1), int(2*text_scale), cv2.LINE_AA)
        rot_ts_frame=cv2.warpAffine(ts_frame,ts_rotation_matrix,(ts_frame.shape[1],ts_frame.shape[0]))[0:frame.shape[0],0:frame.shape[1]]
        rot_ts_frame1=cv2.warpAffine(ts_frame1,ts_rotation_matrix,(ts_frame.shape[1],ts_frame.shape[0]))[0:frame.shape[0],0:frame.shape[1]]
        
        frame[np.where((rot_ts_frame1 != [0,0,0]).all(axis=2))]=[0,0,0]
        frame=cv2.add(frame,rot_ts_frame)
        return frame
        #print(rot_ts_frame.shape,self.first_frame.shape)
        #frame1=np.where(rot_ts_frame==[0,0,0],rot_ts_frame)
        #cv2.imshow("Test",frame)
        #cv2.imshow("Test2",rot_ts_frame)
    

    
    def add_timestamp(self):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
        if cached_frames is not None:
            #cv2.setWindowProperty("Video", cv2.WND_PROP_TOPMOST, 1)
            cv2.setMouseCallback("Video", self.select_timestamp_position)
            cv2.setWindowTitle("Video","ZEITSTEMPEL POSITIONIEREN (KLICK INS BILD)")
            self.reset_all_buttons()
            if platform.system() == "Darwin":   ### if its a Mac
                self.add_time_button.config(relief='sunken',highlightbackground='green')
            else:
                self.add_time_button.config(relief='sunken',bg='green')
            #timestamp_added=True
            

    def select_timestamp_position(self,event, x, y, flags, param):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added, timestamp_position
        if event == cv2.EVENT_LBUTTONDOWN:
            timestamp_position = (x, y)
            timestamp_added=True            
            self.show_first_frame()

   
    def select_kosys(self):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
        if self.kosys.get():
            self.scale.set(False)
            #self.bg.set(False)
        else:
            self.scale.set(True)
        if cached_frames is not None and self.start_point is not None and self.lend_point is not None:
            self.insert_scale()
        
    def select_scale(self):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
        if self.scale.get():
            self.kosys.set(False)
            #self.bg.set(True)
        else:
            self.kosys.set(True)
        if cached_frames is not None and self.start_point is not None and self.lend_point is not None:
            self.insert_scale()
            
    def select_timeshift(self):
        if self.timeshift.get():
            self.timescale.set(False)
            #self.bg.set(False)
        else:
            self.timescale.set(True)
            
    def select_timescale(self):
        if self.timescale.get():
            self.timeshift.set(False)
            #self.bg.set(False)
        else:
            self.timeshift.set(True)
        
            
    def change_alpha(self,value):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
        if self.first_frame is not None:
            if scale_added:
                self.show_first_frame()
                
    
    def return_was_hit(self,event):  
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
        newfps=False
        
        self.frame_max_label.focus_set()

        if cached_frames is not None:
            if float(self.cal.get())!=self.lastcal and lnorm != -1 and self.polar:
                rmax=np.round(self.radius_screen/lnorm*float(self.cal.get()),2)
                self.kleinster_wert_entry.delete(0,tk.END)
                self.kleinster_wert_entry.insert(0,f"0.,{rmax}")
                self.lastcal=float(self.cal.get())
            if float(self.cal.get())!=self.lastcal and lnorm != -1 and (not self.polar):
                self.set_scale_values()
                self.lastcal=float(self.cal.get())
                self.anheft=self.lower_marker.get() 
            if self.lower_marker.get()!=self.anheft and (not self.polar):
                self.set_scale_values()
                self.lastcal=float(self.cal.get())
                self.anheft=self.lower_marker.get() 
            if self.fps_new_entry.get() !="":
                newfps=True
                fps_old=fps
                fps=round(float(self.fps_new_entry.get()),2)
                self.fps_aktuell=fps
                #t0_time=t0_time*fps_old/fps
                t0_time=t0_frame/fps-(t0_frame/fps_old-t0_time)
                self.t0=t0_time
                #fps=float(self.fps_new_entry.get())
                self.fps_new_entry.delete(0,"end")
            #elif fps is None:
                #self.fps=self.cap.get(cv2.CAP_PROP_FPS)
             #   fps=round(fps_native,2)
            #else:
            #    fps=self.fps_aktuell
                #fps=self.fps
                natfps=round(fps_native,2)
            #self.framenum_label.config(text=f"Framenummer: {current_frame}")
                self.framerate_label.config(text=f"{fps}")
                self.native_framerate_label.config(text=f"{natfps}")
            if not imageflag:
                time_diff = current_frame/fps+self.t0_rest
                ten_minutes=int(time_diff/600.)
                time_diff=time_diff-float(ten_minutes)*600.
                minutes=int(time_diff/60.)
                time_diff=time_diff-float(minutes)*60.
                ten_seconds=int(time_diff/10.)
                time_diff=time_diff-float(ten_seconds)*10.
                seconds=int(time_diff)
                time_diff=time_diff-float(seconds)*1.0
                zehntel=int(time_diff/0.1)
                time_diff=time_diff-float(zehntel)*0.1
                hstel=int(time_diff/.01)
                time_diff=time_diff-float(hstel)*0.01
                tstel=int(time_diff/.001)
                timestamp_text = f"Time:  {str(ten_minutes)}{str(minutes)}:{str(ten_seconds)}{str(seconds)}.{str(zehntel)}{str(hstel)}{str(tstel)}"
                timetext=f"{str(ten_minutes)}{str(minutes)}:{str(ten_seconds)}{str(seconds)}.{str(zehntel)}{str(hstel)}{str(tstel)}"
        
                self.time_label.config(
                    text=f"t ab Frame 1: {timetext}")

                #self.t0_frame_label.config(text=f"t={self.t0_rest} bei Frame: {t0_frame}")
                #time_diff=(current_frame - t0_frame) / fps+self.t0_rest
                time_diff=current_frame/fps-self.t0
                time_diff_0=time_diff
                if time_diff<0:
                    negativ=True
                    time_diff=-1.*time_diff
                else:
                    negativ=False
                ten_minutes=int(time_diff/600.)
                time_diff=time_diff-float(ten_minutes)*600.
                minutes=int(time_diff/60.)
                time_diff=time_diff-float(minutes)*60.
                ten_seconds=int(time_diff/10.)
                time_diff=time_diff-float(ten_seconds)*10.
                seconds=int(time_diff)
                time_diff=time_diff-float(seconds)*1.0
                zehntel=int(time_diff/0.1)
                time_diff=time_diff-float(zehntel)*0.1
                hstel=int(time_diff/.01)
                time_diff=time_diff-float(hstel)*0.01
                tstel=int(time_diff/.001)
                if not negativ:
                    timestamp_text = f"Time:  {str(ten_minutes)}{str(minutes)}:{str(ten_seconds)}{str(seconds)}.{str(zehntel)}{str(hstel)}{str(tstel)}"
                    frame_time=f"{str(ten_minutes)}{str(minutes)}:{str(ten_seconds)}{str(seconds)}.{str(zehntel)}{str(hstel)}{str(tstel)}"

                else:
                    timestamp_text = f"Time: -{str(ten_minutes)}{str(minutes)}:{str(ten_seconds)}{str(seconds)}.{str(zehntel)}{str(hstel)}{str(tstel)}"
                    frame_time=f"-{str(ten_minutes)}{str(minutes)}:{str(ten_seconds)}{str(seconds)}.{str(zehntel)}{str(hstel)}{str(tstel)}"

                
                self.time_since_t0_label.config(
                    text=f"t ab t=0: {frame_time}")

                #if self.t1_entry.get() !="" and not newfps and current_frame!=0:
                #    self.set_t1() 

                if newfps:
                    newfps=False


                else:
                    if self.start_point is not None and self.lend_point is not None:
                        self.insert_scale()
                    
            
    
    def redraw(self):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
        newfps=False
        if cached_frames is not None:
            if float(self.cal.get())!=self.lastcal and lnorm != -1 and (not self.polar):
                self.set_scale_values()
                self.lastcal=float(self.cal.get())
            if self.fps_new_entry.get() !="":
                newfps=True
                fps=round(float(self.fps_new_entry.get()),2)
                self.fps_aktuell=fps
                #fps=float(self.fps_new_entry.get())
                self.fps_new_entry.delete(0,"end")
            #elif fps is None:
                #self.fps=self.cap.get(cv2.CAP_PROP_FPS)
             #   fps=round(fps_native,2)
            #else:
            #    fps=self.fps_aktuell
                #fps=self.fps
            natfps=round(fps_native,2)
            #self.framenum_label.config(text=f"Framenummer: {current_frame}")
            self.framerate_label.config(text=f"{fps}")
            self.native_framerate_label.config(text=f"{natfps}")
            
            if not imageflag:
                time_diff = current_frame/fps+self.t0_rest
                ten_minutes=int(time_diff/600.)
                time_diff=time_diff-float(ten_minutes)*600.
                minutes=int(time_diff/60.)
                time_diff=time_diff-float(minutes)*60.
                ten_seconds=int(time_diff/10.)
                time_diff=time_diff-float(ten_seconds)*10.
                seconds=int(time_diff)
                time_diff=time_diff-float(seconds)*1.0
                zehntel=int(time_diff/0.1)
                time_diff=time_diff-float(zehntel)*0.1
                hstel=int(time_diff/.01)
                time_diff=time_diff-float(hstel)*0.01
                tstel=int(time_diff/.001)
                timestamp_text = f"Time:  {str(ten_minutes)}{str(minutes)}:{str(ten_seconds)}{str(seconds)}.{str(zehntel)}{str(hstel)}{str(tstel)}"
                timetext=f"{str(ten_minutes)}{str(minutes)}:{str(ten_seconds)}{str(seconds)}.{str(zehntel)}{str(hstel)}{str(tstel)}"
        
                self.time_label.config(
                    text=f"t ab Frame 1: {timetext}")

                #self.t0_frame_label.config(text=f"t=0 bei Frame: {t0_frame}")
                #time_diff=(current_frame - t0_frame) / fps+self.t0_rest
                time_diff=current_frame/fps-self.t0
                time_diff_0=time_diff
                if time_diff<0:
                    negativ=True
                    time_diff=-1.*time_diff
                else:
                    negativ=False
                ten_minutes=int(time_diff/600.)
                time_diff=time_diff-float(ten_minutes)*600.
                minutes=int(time_diff/60.)
                time_diff=time_diff-float(minutes)*60.
                ten_seconds=int(time_diff/10.)
                time_diff=time_diff-float(ten_seconds)*10.
                seconds=int(time_diff)
                time_diff=time_diff-float(seconds)*1.0
                zehntel=int(time_diff/0.1)
                time_diff=time_diff-float(zehntel)*0.1
                hstel=int(time_diff/.01)
                time_diff=time_diff-float(hstel)*0.01
                tstel=int(time_diff/.001)
                if not negativ:
                    timestamp_text = f"Time:  {str(ten_minutes)}{str(minutes)}:{str(ten_seconds)}{str(seconds)}.{str(zehntel)}{str(hstel)}{str(tstel)}"
                    frame_time=f"{str(ten_minutes)}{str(minutes)}:{str(ten_seconds)}{str(seconds)}.{str(zehntel)}{str(hstel)}{str(tstel)}"

                else:
                    timestamp_text = f"Time: -{str(ten_minutes)}{str(minutes)}:{str(ten_seconds)}{str(seconds)}.{str(zehntel)}{str(hstel)}{str(tstel)}"
                    frame_time=f"-{str(ten_minutes)}{str(minutes)}:{str(ten_seconds)}{str(seconds)}.{str(zehntel)}{str(hstel)}{str(tstel)}"

                
                self.time_since_t0_label.config(
                    text=f"t ab t=0: {frame_time}")

                if self.t1_entry.get() !="" and current_frame!=0:
                    self.set_t1() 

            if newfps:
                newfps=False


            else:
                if self.start_point is not None and self.lend_point is not None:
                    self.insert_scale()
        
    def white_bg(self):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
        if cached_frames is not None and self.start_point is not None and self.lend_point is not None:
            self.insert_scale()
        
    def inverty(self):
        if scale_added:
            self.insert_scale()
        else:
            self.show_first_frame()
        
            
    
    def open_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if file_path:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(file_path)
            current_frame=0           
            width_px = root.winfo_screenwidth()
            height_px = int(0.9*root.winfo_screenheight())
            #print('HEIGHT',height_px) 
            #if self.first_frame is not None:
             #   if self.first_frame.shape[0]>height_px and platform.system() != "Darwin":
              #      tempimage = self.ResizeWithAspectRatio(self.first_frame, height=height_px)
               #     self.first_frame=tempimage
            
            #self.show_first_frame()
            #cv2.waitKey(5000)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            ret,frame = self.cap.read()
            if ret:
                if frame.shape[0]>height_px and platform.system() != "Darwin":
                    tempimage = self.ResizeWithAspectRatio(frame, height=height_px)
                    frame=tempimage 
                self.cached_frames = {0:frame.copy()}

                text='VIDEO WIRD GELADEN!'
                font = cv2.FONT_HERSHEY_DUPLEX
                #imgtext=self.first_frame
                textsize = cv2.getTextSize(text, font, 3, 2)[0]
                
                textX = int((frame.shape[1] - textsize[0]) / 2)
                textY = int((frame.shape[0] + textsize[1]) / 2 )
                
                #print('TEXT',textX,textY,text)
                
                cv2.putText(frame, text, (textX,textY),font, 3, (0, 0, 255), 2,cv2.LINE_AA)
                cv2.imshow("Video", frame)
                #cv2.waitKey(1)
                x=self.root.winfo_x()
                y=self.root.winfo_y()
                w=self.root.winfo_width()
                cv2.moveWindow("Video",x+w+30,y)
                cv2.imshow("Video", frame)
                cv2.waitKey(1)

                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.video_slider.config(to=self.total_frames - 1, resolution=1)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES,1)
                # Lese die Frames ein und speichere sie im Cache-Dictionary
                for i in range(1,self.total_frames-1):
                    #self.cap.set(cv2.CAP_PROP_POS_FRAMES,i)
                    ret, frame = self.cap.read()
                    #print(self.total_frames,total_frames,i,self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if frame is not None:
                        if frame.shape[0]>height_px and platform.system() != "Darwin":
                            tempimage = self.ResizeWithAspectRatio(frame, height=height_px)
                            frame=tempimage  
                        self.cached_frames[i]=frame.copy()
                        if(i%10==0):
                            frame=self.cached_frames[i].copy()
                            cv2.putText(frame, text, (textX,textY),font, 3, (0, 0, 255), 2,cv2.LINE_AA)
                            cv2.imshow("Video", frame)
                            cv2.waitKey(1)
                    else:
                        self.total_frames=i-1
                        messagebox.showerror("Fehler beim Einlesen des Videos",f"Abbruch bei Frame {i-1}")
                        self.video_slider.config(to=self.total_frames - 1, resolution=1)
                        break
                print("cached")
                current_frame=0
                fps=self.cap.get(cv2.CAP_PROP_FPS)
                self.first_frame = cached_frames[0].copy()
                self.show_first_frame()
                self.mark_start()

    def show_first_frame(self):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
        global imageflag
        self.Plotting=True
        if cached_frames is not None and current_frame in cached_frames:
            #self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            #ret, self.first_frame = self.cap.read()
            #if ret:
            #    if self.first_frame is not None:
            #        height_px = int(0.9*root.winfo_screenheight())
            #        if self.first_frame.shape[0]>height_px and platform.system() != "Darwin":
            #            tempimage = self.ResizeWithAspectRatio(self.first_frame, height=height_px)
            self.first_frame=cached_frames[current_frame].copy()
            
            natfps=round(fps_native,2)
            #self.framenum_label.config(text=f"Framenummer: {current_frame}")
            self.framerate_label.config(text=f"{fps}")
            self.native_framerate_label.config(text=f"{natfps}")

            if not imageflag:
                self.update_control_panel(total_frames,fps)
            if timestamp_added:
                #self.draw_timestamp(self.first_frame)
                frame2=self.first_frame.copy()
                frame1=self.draw_timestamp(frame2)
                self.first_frame=frame1.copy()

                #cv2.imshow("Test",self.first_frame)
                #print('timestamp added')
                


            if self.start_end_selected_crop:
                    
                    #cv2.circle(self.first_frame,(x1,y1), 5, (0,0,255), -1)
                    #cv2.circle(self.first_frame,(x2,y2), 5, (0,0,255), -1)
                    cv2.rectangle(self.first_frame,(self.crop_x0,self.crop_y0),(self.crop_x1,self.crop_y1),(0,120,255),3)
                
            
            if self.l_selected:
                x1,y1=self.lstart_point
                x2,y2=self.lend_point
                #cv2.circle(self.first_frame,(x1,y1), 5, (0,0,255), -1)
                #cv2.circle(self.first_frame,(x2,y2), 5, (0,0,255), -1)
                cv2.line(self.first_frame,(x1,y1),(x2,y2),(0,0,255),3)
            if self.start_end_selected:
                        x1,y1=self.start_point
                        x2,y2=self.end_point
                        #cv2.circle(self.first_frame,(x1,y1), 5, (255,0,0), -1)
                        #cv2.circle(self.first_frame,(x2,y2), 5, (255,0,0), -1)
                        tl=15./max(np.sqrt((x1-x2)**2.+(y1-y2)**2.),.01)
                        dx=x2-x1
                        dy=y2-y1
                        if x2>x1:
                            if self.invertvar_y.get():
                                yy=y1+dx
                                xx=x1-dy
                            else:
                                yy=y1-dx
                                xx=x1+dy
                        else:
                            if self.invertvar_y.get():
                                yy=y1-dx
                                xx=x1+dy
                            else:
                                yy=y1+dx
                                xx=x1-dy
                        if self.kosys.get() and not self.polar:
                            cv2.arrowedLine(self.first_frame,(x1,y1),(xx,yy),(0,255,0),3,tipLength=tl)
                        cv2.arrowedLine(self.first_frame,(x1,y1),(x2,y2),(255,0,0),3,tipLength=tl)
            if scale_added:
                self.overlay_image(self.first_frame)
                
            cv2.imshow("Video", self.first_frame)
            
            self.Plotting=False
            
            

    def set_frame(self, value):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
        if not self.Plotting:
            if cached_frames is not None and int(value) in cached_frames:
                current_frame = int(value)
                #self.first_frame=cached_frames[current_frame].copy()
                #self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                #ret, frame = self.cap.read()
                #if ret:
                    #self.first_frame=frame          
                self.show_first_frame()
                
    def next_frame(self):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
        if cached_frames is not None and self.video_slider.get()+1 in cached_frames:
        
         
            current_frame = self.video_slider.get()+1
            self.video_slider.set(current_frame)
            #self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            #ret, frame = self.cap.read()
            #if ret:
                #self.first_frame=frame           
            self.show_first_frame()
            

    def prev_frame(self):
         global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
         if cached_frames is not None and self.video_slider.get()-1 in cached_frames:
            current_frame = self.video_slider.get()-1
            self.video_slider.set(current_frame)
                #self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                #ret, frame = self.cap.read()
                #if ret:
                    #self.first_frame=frame           
            self.show_first_frame()

    def select_polarmethod(self):
        self.polarmode_select = tk.Toplevel()
        self.polarmode_select.title("Modus wählen")
        self.three_point_method = tk.Button(self.polarmode_select, text="Drei Punkte auf Kreis \n wählen", width=20, height=3, command=lambda: self.mark_polarpoints("ThreePoints"))
        self.three_point_method.grid(row=0,column=1,pady=5, rowspan=2,columnspan=3)
        self.center_radius_method = tk.Button(self.polarmode_select, text="Mittelpunkt und Radius \n wählen", width=20, height=3,command=lambda: self.mark_polarpoints("CenterRadius"))
        self.center_radius_method.grid(row=0,column=4,pady=5, rowspan=2, columnspan=3)

    def mark_polarpoints(self,polarmode):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
    
        if self.first_frame is not None and polarmode is not None:
            #print('Hallo')
            #self.show_first_frame()
            #x=self.root.winfo_x()
            #y=self.root.winfo_y()
            #w=self.root.winfo_width()
            #cv2.moveWindow("Video",x+w+30,y)
            #self.show_first_frame()
            #cv2.waitKey(1)
            self.polar=True
            self.polaerscale_deg_button.config(state="active")
            self.polaerscale_rad_button.config(state="active")
            self.polaerscale_length_button.config(state="active")

            self.kleinster_wert_label.config(text=f"r_min,r_max:")
            self.ybounds_label.config(text=f"theta_min, theta_max:")
            self.ybounds_entry.delete(0,tk.END)
            self.ybounds_entry.insert(0,'0.0 , 360.0')
            self.labels_label_entry.delete(0,tk.END)
            self.labels_label_entry.insert(0, '45')
            self.longticks_entry.delete(0,tk.END)
            self.longticks_entry.insert(0,'10')
            self.shortticks_entry.delete(0,tk.END)
            self.shortticks_entry.insert(0,'5')
            self.zerolabel_button.config(text="Label bei erstem Wert")
            self.zerolabel.set(True)
            self.lastlabelvar.set(False)

            self.show_ylabels.set(False)
            self.first_circle=True
        
            

            #cv2.setWindowProperty("Video", cv2.WND_PROP_TOPMOST, 1)
            if polarmode=="ThreePoints":
                cv2.setMouseCallback("Video", self.select_polar)
            if polarmode=="CenterRadius":
                cv2.setMouseCallback("Video", self.select_polar_center_radius)
            self.polarmode_select.destroy()
            #cv2.setWindowTitle("Video","3 Punkte auf dem Kreis markieren")
            self.reset_all_buttons()
            if platform.system() == "Darwin":   ### if its a Mac
                self.polar_button.config(relief='sunken',highlightbackground='pink')
            else:
                self.start_button.config(relief='sunken',bg='pink')
    
    def select_polar_center_radius(self, event, x, y, flags, param):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
        global start_x,start_y,end_x,end_y,scale_length,scale_image,x_shift,y_shift,pos_x,pos_y,lnorm, lstart_x, lstart_y

        if event == cv2.EVENT_MOUSEMOVE:
            #self.coordinates_label.config(text=f"Pixel-Koordinaten: ({x}, {y})")
            
            cv2.setWindowTitle("Video",f"P({x}, {y})")
            if self.selecting_polar_radius:
            #if not self.start_end_selected:
                if 1==1:
                    #self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                    #ret, self.first_frame = self.cap.read()
                    self.first_frame = cached_frames[current_frame].copy()
                    if 1==1:
                        #height_px = int(0.9*root.winfo_screenheight())
                        #if self.first_frame.shape[0]>height_px and platform.system() != "Darwin":
                        #    tempimage = self.ResizeWithAspectRatio(self.first_frame, height=height_px)
                        #    self.first_frame=tempimage
                        if self.l_selected:
                            x1,y1=self.lstart_point
                            x2,y2=self.lend_point
                            #cv2.circle(self.first_frame,(x1,y1), 5, (0,0,255), -1)
                            #cv2.circle(self.first_frame,(x2,y2), 5, (0,0,255), -1)
                            cv2.line(self.first_frame,(x1,y1),(x2,y2),(0,0,255),3)
                        if self.start_end_selected:
                            x1,y1=self.start_point
                            x2,y2=self.end_point
                            #cv2.circle(self.first_frame,(x1,y1), 5, (255,0,0), -1)
                            #cv2.circle(self.first_frame,(x2,y2), 5, (255,0,0), -1)
                            #cv2.arrowedLine(self.first_frame,(x1,y1),(x2,y2),(255,0,0),3,tipLength=0.04)
                        #if self.scale_added:
                         #   self.overlay_image(self.first_frame)
                        if timestamp_added:
                            frame1=self.draw_timestamp(self.first_frame)
                            self.first_frame=frame1.copy()
                        
                        cv2.circle(self.first_frame,self.polar_center , 5, (255,0,0), -1)
                        cx,cy=self.polar_center
                        r=int(round(np.sqrt((x-cx)**2.0+(y-cy)**2.0),0))
                        cv2.circle(self.first_frame,self.polar_center,r,(255,0,0),2)
                        cv2.imshow("Video", self.first_frame)
                        #cv2.waitKey(1)
                        #print("HUUUUUHHHUUUUUUUUUUUUUUUUU",r)
            if self.selecting_polar_end:
                #if not self.start_end_selected:
                if 1==1:
                    #self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                    #ret, self.first_frame = self.cap.read()
                    self.first_frame = cached_frames[current_frame].copy()
                    if 1==1:
                        #height_px = int(0.9*root.winfo_screenheight())
                        #if self.first_frame.shape[0]>height_px and platform.system() != "Darwin":
                        #    tempimage = self.ResizeWithAspectRatio(self.first_frame, height=height_px)
                        #    self.first_frame=tempimage
                        if self.l_selected:
                            x1,y1=self.lstart_point
                            x2,y2=self.lend_point
                            #cv2.circle(self.first_frame,(x1,y1), 5, (0,0,255), -1)
                            #cv2.circle(self.first_frame,(x2,y2), 5, (0,0,255), -1)
                            cv2.line(self.first_frame,(x1,y1),(x2,y2),(0,0,255),3)
                        if self.start_end_selected:
                            x1,y1=self.start_point
                            x2,y2=self.end_point
                            #cv2.circle(self.first_frame,(x1,y1), 5, (255,0,0), -1)
                            #cv2.circle(self.first_frame,(x2,y2), 5, (255,0,0), -1)
                            #cv2.arrowedLine(self.first_frame,(x1,y1),(x2,y2),(255,0,0),3,tipLength=0.04)
                        #if self.scale_added:
                         #   self.overlay_image(self.first_frame)
                        if timestamp_added:
                            frame1=self.draw_timestamp(self.first_frame)
                            self.first_frame=frame1.copy()
                        
                        cv2.circle(self.first_frame,self.polar_center , 5, (255,0,0), -1)
                        cv2.circle(self.first_frame,self.polar_center,self.radius_screen,(255,0,0),2)

                        tl=15./max(np.sqrt((start_x-x)**2.+(start_y-y)**2.),.01)
                        
                        dx=x-start_x
                        dy=y-start_y
                        if x>start_x:
                            if self.invertvar_y.get():
                                yy=start_y+dx
                                xx=start_x-dy
                            else:
                                yy=start_y-dx
                                xx=start_x+dy
                        else:
                            if self.invertvar_y.get():
                                yy=start_y-dx
                                xx=start_x+dy
                            else:
                                yy=start_y+dx
                                xx=start_x-dy
                                
                        cv2.arrowedLine(self.first_frame,(start_x,start_y),(x,y),(255,0,0),3,tipLength=tl)
                        #if self.kosys.get():
                         #   cv2.arrowedLine(self.first_frame,(start_x,start_y),(xx,yy),(0,255,0),3,tipLength=tl)
                      #  self.first_frame=self.cached_frames[current_frame]
                        cv2.imshow("Video", self.first_frame)
                        #cv2.waitKey(1)
                #else:
                #    self.start_end_selected=False
        
                
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.selecting_polar_center:
                self.polar_center = (x, y)
                #Ausgewählten Punkt anzeigen
                self.first_frame = cached_frames[current_frame].copy()
                cv2.circle(self.first_frame,self.polar_center , 5, (255,0,0), -1)
                cv2.imshow("Video", self.first_frame)
                self.selecting_polar_center = False
                self.selecting_polar_radius = True
                #self.start_point_label.config(text=f"Startpunkt: ({x}, {y})")
            
            elif self.selecting_polar_radius:
                self.selecting_polar_radius=False
                self.selecting_polar_end = True
                
                center_x,center_y=self.polar_center
                self.radius_screen=int(round(np.sqrt((center_x-x)**2.+(center_y-y)**2.),0))

                self.first_frame = cached_frames[current_frame].copy()
                cv2.circle(self.first_frame,self.polar_center , 5, (255,0,0), -1)
                cv2.circle(self.first_frame,self.polar_center,self.radius_screen,(255,0,0),2)
                cv2.imshow("Video", self.first_frame)
                self.polar_circle=True

                self.start_point=self.polar_center
                start_x,start_y=self.start_point

            elif self.selecting_polar_end:
                self.end_point = (x, y)
                self.end_point_bak = (x,y)
                end_x,end_y=self.end_point
                self.selecting_polar_end=False
                self.selecting_polar_center=True
                #if start_x > end_x:
                    #self.invertx=True
               # else:
                 #   self.invertx=False
                self.start_end_selected=True

                ### Länge Pfeilspitze an Pfeillänge anpassen
                tl=15./max(np.sqrt((start_x-x)**2.+(start_y-y)**2.),.01)        
                dx=x-start_x
                dy=y-start_y
                if x>start_x:
                    if self.invertvar_y.get():
                        yy=start_y+dx
                        xx=start_x-dy
                    else:
                        yy=start_y-dx
                        xx=start_x+dy
                else:
                    if self.invertvar_y.get():
                        yy=start_y-dx
                        xx=start_x+dy
                    else:
                        yy=start_y+dx
                        xx=start_x-dy
                        
                cv2.arrowedLine(self.first_frame,(start_x,start_y),(x,y),(255,0,0),3,tipLength=tl)
                #cv2.arrowedLine(self.first_frame,(start_x,start_y),(xx,yy),(0,255,0),3,tipLength=tl)
                #  self.first_frame=self.cached_frames[current_frame]
                cv2.imshow("Video", self.first_frame)
                self.mark_length()

    def select_polar(self, event, x, y, flags, param):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
        global start_x,start_y,end_x,end_y,scale_length,scale_image,x_shift,y_shift,pos_x,pos_y,lnorm, lstart_x, lstart_y
        
        if event == cv2.EVENT_MOUSEMOVE:
            #self.coordinates_label.config(text=f"Pixel-Koordinaten: ({x}, {y})")
            
            cv2.setWindowTitle("Video",f"P({x}, {y})")
            if self.selecting_polar_end:
                #if not self.start_end_selected:
                if 1==1:
                    #self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                    #ret, self.first_frame = self.cap.read()
                    self.first_frame = cached_frames[current_frame].copy()
                    if 1==1:
                        #height_px = int(0.9*root.winfo_screenheight())
                        #if self.first_frame.shape[0]>height_px and platform.system() != "Darwin":
                        #    tempimage = self.ResizeWithAspectRatio(self.first_frame, height=height_px)
                        #    self.first_frame=tempimage
                        if self.l_selected:
                            x1,y1=self.lstart_point
                            x2,y2=self.lend_point
                            #cv2.circle(self.first_frame,(x1,y1), 5, (0,0,255), -1)
                            #cv2.circle(self.first_frame,(x2,y2), 5, (0,0,255), -1)
                            cv2.line(self.first_frame,(x1,y1),(x2,y2),(0,0,255),3)
                        if self.start_end_selected:
                            x1,y1=self.start_point
                            x2,y2=self.end_point
                            #cv2.circle(self.first_frame,(x1,y1), 5, (255,0,0), -1)
                            #cv2.circle(self.first_frame,(x2,y2), 5, (255,0,0), -1)
                            #cv2.arrowedLine(self.first_frame,(x1,y1),(x2,y2),(255,0,0),3,tipLength=0.04)
                        #if self.scale_added:
                            
                        if timestamp_added:
                            frame1=self.draw_timestamp(self.first_frame)
                            self.first_frame=frame1.copy()

                         #   self.overlay_image(self.first_frame)
                        cv2.circle(self.first_frame,self.polar1 , 5, (255,0,0), -1)
                        cv2.circle(self.first_frame,self.polar2 , 5, (255,0,0), -1)
                        cv2.circle(self.first_frame,self.polar3 , 5, (255,0,0), -1)
                        cv2.circle(self.first_frame,self.polar_center , 5, (255,0,0), -1)
                        cv2.circle(self.first_frame,self.polar_center,self.radius_screen,(255,0,0),2)

                        tl=15./max(np.sqrt((start_x-x)**2.+(start_y-y)**2.),.01)
                        
                        dx=x-start_x
                        dy=y-start_y
                        if x>start_x:
                            if self.invertvar_y.get():
                                yy=start_y+dx
                                xx=start_x-dy
                            else:
                                yy=start_y-dx
                                xx=start_x+dy
                        else:
                            if self.invertvar_y.get():
                                yy=start_y-dx
                                xx=start_x+dy
                            else:
                                yy=start_y+dx
                                xx=start_x-dy
                                
                        cv2.arrowedLine(self.first_frame,(start_x,start_y),(x,y),(255,0,0),3,tipLength=tl)
                        #if self.kosys.get():
                         #   cv2.arrowedLine(self.first_frame,(start_x,start_y),(xx,yy),(0,255,0),3,tipLength=tl)
                      #  self.first_frame=self.cached_frames[current_frame]
                        cv2.imshow("Video", self.first_frame)
                        #cv2.waitKey(1)
                #else:
                #    self.start_end_selected=False
        
                
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.selecting_first_polar:
                self.polar1 = (x, y)
                #Ausgewählten Punkt anzeigen
                self.first_frame = cached_frames[current_frame].copy()
                cv2.circle(self.first_frame,self.polar1 , 5, (255,0,0), -1)
                cv2.imshow("Video", self.first_frame)
                self.selecting_first_polar = False
                self.selecting_second_polar = True
                #self.start_point_label.config(text=f"Startpunkt: ({x}, {y})")
            elif self.selecting_second_polar:
                self.polar2 = (x, y)
                self.selecting_second_polar=False
                self.selecting_third_polar = True
                self.first_frame = cached_frames[current_frame].copy()
                cv2.circle(self.first_frame,self.polar1 , 5, (255,0,0), -1)
                cv2.circle(self.first_frame,self.polar2 , 5, (255,0,0), -1)
                cv2.imshow("Video", self.first_frame)
            elif self.selecting_third_polar:
                self.polar3 = (x, y)
                self.selecting_third_polar=False
                self.selecting_polar_end = True
                
                x1,y1=self.polar1
                x2,y2=self.polar2
                x3,y3=self.polar3
                K1=-(x1**2.+y1**2.)
                K2=-(x2**2.+y2**2.)
                K3=-(x3**2.+y3**2.)
                coeff_mat=[[1,-x1,-y1],[1,-x2,-y2],[1,-x3,-y3]]
                b=[K1,K2,K3]
                ABC=np.linalg.solve(coeff_mat,b)
                
                center_x=ABC[1]/2.
                center_y=ABC[2]/2.
                self.polar_center=(int(round(center_x,0)),int(round(center_y,0)))
                self.radius_screen=int(round(np.sqrt(center_x**2.+center_y**2.-ABC[0]),0))

                self.first_frame = cached_frames[current_frame].copy()
                cv2.circle(self.first_frame,self.polar1 , 5, (255,0,0), -1)
                cv2.circle(self.first_frame,self.polar2 , 5, (255,0,0), -1)
                cv2.circle(self.first_frame,self.polar3 , 5, (255,0,0), -1)
                cv2.circle(self.first_frame,self.polar_center , 5, (255,0,0), -1)
                cv2.circle(self.first_frame,self.polar_center,self.radius_screen,(255,0,0),2)
                cv2.imshow("Video", self.first_frame)
                self.polar_circle=True

                self.start_point=self.polar_center
                start_x,start_y=self.start_point

            elif self.selecting_polar_end:
                self.end_point = (x, y)
                self.end_point_bak = (x, y)
                end_x,end_y=self.end_point
                self.selecting_polar_end=False
                self.selecting_first_polar=True
                #if start_x > end_x:
                    #self.invertx=True
               # else:
                 #   self.invertx=False
                self.start_end_selected=True

                ### Länge Pfeilspitze an Pfeillänge anpassen
                tl=15./max(np.sqrt((start_x-x)**2.+(start_y-y)**2.),.01)        
                dx=x-start_x
                dy=y-start_y
                if x>start_x:
                    if self.invertvar_y.get():
                        yy=start_y+dx
                        xx=start_x-dy
                    else:
                        yy=start_y-dx
                        xx=start_x+dy
                else:
                    if self.invertvar_y.get():
                        yy=start_y-dx
                        xx=start_x+dy
                    else:
                        yy=start_y+dx
                        xx=start_x-dy
                        
                cv2.arrowedLine(self.first_frame,(start_x,start_y),(x,y),(255,0,0),3,tipLength=tl)
               # cv2.arrowedLine(self.first_frame,(start_x,start_y),(xx,yy),(0,255,0),3,tipLength=tl)
                #  self.first_frame=self.cached_frames[current_frame]
                cv2.imshow("Video", self.first_frame)
                self.mark_length()
    
                
                                        

    def mark_start(self):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
        if self.first_frame is not None:
            #print('Hallo')
            #self.show_first_frame()
            #x=self.root.winfo_x()
            #y=self.root.winfo_y()
            #w=self.root.winfo_width()
            #cv2.moveWindow("Video",x+w+30,y)
            #self.show_first_frame()
            #cv2.waitKey(1)
            self.polar=False
            #cv2.setWindowProperty("Video", cv2.WND_PROP_TOPMOST, 1)
            cv2.setMouseCallback("Video", self.select_point)
            cv2.setWindowTitle("Video","KO-SYSTEM AUSRICHTEN (klicken, ziehen, klicken)")
            self.reset_all_buttons()
            if platform.system() == "Darwin":   ### if its a Mac
                self.start_button.config(relief='sunken',highlightbackground='blue')
            else:
                self.start_button.config(relief='sunken',bg='blue')
                

    def select_point(self, event, x, y, flags, param):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
        global start_x,start_y,end_x,end_y,scale_length,scale_image,x_shift,y_shift,pos_x,pos_y,lnorm, lstart_x, lstart_y
        if event == cv2.EVENT_MOUSEMOVE:
            #self.coordinates_label.config(text=f"Pixel-Koordinaten: ({x}, {y})")
            if end_x!=-1:
                cv2.setWindowTitle("Video",f"START({start_x} , {start_y})   ENDE({start_y}, {end_y})   P({x}, {y}) ")
            else:
                if start_x!=-1:
                    cv2.setWindowTitle("Video",f"START({start_x}, {start_y})   P({x}, {y})")
                else:
                    cv2.setWindowTitle("Video",f"P({x}, {y})")
            if not self.selecting_start_point:
                if not self.start_end_selected:
                    #self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                    #ret, self.first_frame = self.cap.read()
                    self.first_frame = cached_frames[current_frame].copy()
                    if 1==1:
                        #height_px = int(0.9*root.winfo_screenheight())
                        #if self.first_frame.shape[0]>height_px and platform.system() != "Darwin":
                        #    tempimage = self.ResizeWithAspectRatio(self.first_frame, height=height_px)
                        #    self.first_frame=tempimage
                        if self.l_selected:
                            x1,y1=self.lstart_point
                            x2,y2=self.lend_point
                            cv2.circle(self.first_frame,(x1,y1), 5, (0,0,255), -1)
                            cv2.circle(self.first_frame,(x2,y2), 5, (0,0,255), -1)
                            cv2.line(self.first_frame,(x1,y1),(x2,y2),(0,0,255),3)
                        if self.start_end_selected:
                            x1,y1=self.start_point
                            x2,y2=self.end_point
                            #cv2.circle(self.first_frame,(x1,y1), 5, (255,0,0), -1)
                            #cv2.circle(self.first_frame,(x2,y2), 5, (255,0,0), -1)
                            #cv2.arrowedLine(self.first_frame,(x1,y1),(x2,y2),(255,0,0),3,tipLength=0.04)
                        #if self.scale_added:
                         #   self.overlay_image(self.first_frame)
                        tl=15./max(np.sqrt((start_x-x)**2.+(start_y-y)**2.),.01)
                        
                        dx=x-start_x
                        dy=y-start_y
                        if x>start_x:
                            if self.invertvar_y.get():
                                yy=start_y+dx
                                xx=start_x-dy
                            else:
                                yy=start_y-dx
                                xx=start_x+dy
                        else:
                            if self.invertvar_y.get():
                                yy=start_y-dx
                                xx=start_x+dy
                            else:
                                yy=start_y+dx
                                xx=start_x-dy
                                
                        cv2.arrowedLine(self.first_frame,(start_x,start_y),(x,y),(255,0,0),3,tipLength=tl)
                        if self.kosys.get():
                            cv2.arrowedLine(self.first_frame,(start_x,start_y),(xx,yy),(0,255,0),3,tipLength=tl)
                      #  self.first_frame=self.cached_frames[current_frame]
                        cv2.imshow("Video", self.first_frame)
                        #cv2.waitKey(1)
                else:
                    self.start_end_selected=False
            
            
            
            
            
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.selecting_start_point:
                self.start_point = (x, y)
                start_x,start_y=self.start_point
                #Ausgewählten Punkt anzeigen
                self.first_frame = cached_frames[current_frame].copy()
                cv2.circle(self.first_frame,(x,y), 5, (255,0,0), -1)
                cv2.imshow("Video", self.first_frame)
                self.selecting_start_point = False
                #self.start_point_label.config(text=f"Startpunkt: ({x}, {y})")
            else:
                self.end_point = (x, y)
                self.end_point_bak= (x,y)
                self.selecting_start_point = True
                end_x,end_y=self.end_point
                #self.first_frame = self.cached_frames[current_frame].copy()
                #cv2.imshow("Video",self.first_frame)
                #cv2.waitKey(1)
                #Ausgewählten Punkt anzeigen
                #cv2.circle(self.first_frame,(x,y), 5, (255,0,0), -1)
                cv2.arrowedLine(self.first_frame,(start_x,start_y),(x,y),(255,0,0),3,tipLength=0.02)
                cv2.imshow("Video", self.first_frame)
                if start_x > end_x and not self.polar:
                #if start_x > end_x:
                    self.invertx=True
                else:
                    self.invertx=False
                self.start_end_selected=True
                self.mark_length()
                #self.end_point_label.config(text=f"Richtung: ({x}, {y})")

    def mark_length(self):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
        if self.first_frame is not None:
            self.reset_all_buttons()
            if platform.system() == "Darwin":   ### if its a Mac
                self.length_button.config(relief='sunken',highlightbackground='red')
            else:
                self.length_button.config(relief='sunken',bg='red')
            #cv2.setWindowProperty("Video", cv2.WND_PROP_TOPMOST, 1)
            cv2.setMouseCallback("Video", self.select_point_2)
            cv2.setWindowTitle("Video","MASSSTAB FESTLEGEN (klicken, ziehen, klicken)")

    def select_point_2(self, event, x, y, flags, param):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
        global start_x,start_y,end_x,end_y,scale_length,scale_image,x_shift,y_shift,pos_x,pos_y,lnorm, lstart_x, lstart_y
        if event == cv2.EVENT_MOUSEMOVE:
            #self.coordinates_label.config(text=f"Pixel-Koordinaten: ({x}, {y})")
            if self.lend_point is not None:
                x1,y1=self.lstart_point
                x2,y2=self.lend_point
                cv2.setWindowTitle("Video",f"START({x1} , {y1})   ENDE({x2}, {y2 })   P({x}, {y}) ")
            else:
                if self.lstart_point is not None:
                    x1,y1=self.lstart_point
                    cv2.setWindowTitle("Video",f"START({x1}, {y1})   P({x}, {y})")
                else:
                    cv2.setWindowTitle("Video",f"P({x}, {y})")
            #WENN DU BEI DER PUNKTAUSWAHL BIST, ABER NICHT DEN STARTPUNKT WÄHLST (ALSO DEN ENDPUNKT!)
            if not self.selecting_lstart_point:
                if not self.l_selected:
                    #self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                    #ret, self.first_frame = self.cap.read()
                    self.first_frame = cached_frames[current_frame].copy()
                    if 1==1:
                        #height_px = int(0.9*root.winfo_screenheight())
                        #if self.first_frame.shape[0]>height_px and platform.system() != "Darwin":
                        #    tempimage = self.ResizeWithAspectRatio(self.first_frame, height=height_px)
                        #    self.first_frame=tempimage
                        if self.l_selected:
                            x1,y1=self.lstart_point
                            x2,y2=self.lend_point
                            #cv2.circle(self.first_frame,(x1,y1), 5, (0,0,255), -1)
                            #cv2.circle(self.first_frame,(x2,y2), 5, (0,0,255), -1)
                            cv2.line(self.first_frame,(x1,y1),(x2,y2),(0,0,255),3)
                            cv2.waitKey(1)
                        
                        if self.polar_circle:
                            cv2.circle(self.first_frame,self.polar_center , 5, (255,0,0), -1)
                            cv2.circle(self.first_frame,self.polar_center,self.radius_screen,(255,0,0),2)
                        if self.start_end_selected:
                            x1,y1=self.start_point
                            x2,y2=self.end_point
                            #cv2.circle(self.first_frame,(x1,y1), 5, (255,0,0), -1)
                            #cv2.circle(self.first_frame,(x2,y2), 5, (255,0,0), -1)
                            tl=15./max(np.sqrt((x1-x2)**2.+(y1-y2)**2.),.01)
                            dx=x2-x1
                            dy=y2-y1
                            if x2>x1:
                                if self.invertvar_y.get():
                                    yy=y1+dx
                                    xx=x1-dy
                                else:
                                    yy=y1-dx
                                    xx=x1+dy
                            else:
                                if self.invertvar_y.get():
                                    yy=y1-dx
                                    xx=x1+dy
                                else:
                                    yy=y1+dx
                                    xx=x1-dy
                            if self.kosys.get() and not self.polar:
                                cv2.arrowedLine(self.first_frame,(x1,y1),(xx,yy),(0,255,0),3,tipLength=tl)
                            cv2.arrowedLine(self.first_frame,(x1,y1),(x2,y2),(255,0,0),3,tipLength=tl)
                        #if self.scale_added:
                         #   self.overlay_image(self.first_frame)
                        #cv2.circle(self.first_frame,(lstart_x,lstart_y), 5, (0,0,255), -1)  
                        cv2.line(self.first_frame,(lstart_x,lstart_y),(x,y),(0,0,255),3)
                        cv2.imshow("Video", self.first_frame)
                        
                else:
                    self.l_selected=False
            
            
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.selecting_lstart_point:
                self.lstart_point = (x, y)
                #Ausgewählten Punkt anzeigen
                #cv2.circle(self.first_frame,(x,y), 5, (0,0,255), -1)
                cv2.imshow("Video", self.first_frame)
                lstart_x,lstart_y=self.lstart_point
                print(lstart_x,lstart_y)
                self.selecting_lstart_point = False
                #self.start_point_label.config(text=f"Startpunkt: ({x}, {y})")
            else:
                self.lend_point = (x, y)
                self.selecting_lstart_point = True
                lend_x,lend_y=self.lend_point
                #print(lend_x,lend_y)
                #Ausgewählten Punkt anzeigen
                #cv2.circle(self.first_frame,(x,y), 5, (0,0,255), -1)
                cv2.line(self.first_frame,(lstart_x,lstart_y),(x,y),(0,0,255),3)
                cv2.imshow("Video", self.first_frame)
                self.l_selected=True

                #Abstand berechnen
                dx=lend_x-lstart_x
                dy=lend_y-lstart_y
                
                lnorm=np.sqrt(dx**2.+dy**2.)
                if not self.polar:
                    self.set_scale_values()
                
                if self.start_point is not None:
                    self.insert_scale()
                    #self.show_first_frame()
                #print('NORM',lnorm)
                #self.end_point_label.config(text=f"Endpunkt: ({x}, {y})")

    def set_scale_values(self):
        global lnorm
        if lnorm !=-1:
            frame_diag=np.sqrt(self.first_frame.shape[0]**2.+self.first_frame.shape[1]**2.)
            n_scales=frame_diag/lnorm
            print('Nscales',n_scales)
            #print(frame_diag,lnorm,n_scales)

            #Anheftpunkt auslesen:
            markstr=self.lower_marker.get()
            mx,my = map(float,markstr.split(','))


            l=float(self.cal.get())
            x_max=max(round(0.5*n_scales*l,1),l)
            y_max=round(0.75*x_max,1)
            labeldist=x_max/10.

            print('Labeldist',labeldist)

            if labeldist>1:
                order = int(np.log10(labeldist))
            if labeldist<1:
                order = int(np.log10(labeldist))-1.
            if labeldist==0:
                labeldist=.1
                order=-1.


            print('Order',order)
            normed_labeldist=round(labeldist/(10.**order),0)
            print('Normeddist',normed_labeldist)
            if normed_labeldist == 3.:
                normed_labeldist =2.
            if normed_labeldist == 6. or normed_labeldist==7. or normed_labeldist==4.:
                normed_labeldist =5.
            if normed_labeldist == 8. or normed_labeldist==9.:
                normed_labeldist =10.
            labeldist=normed_labeldist*10.**order

            x_max=10.*labeldist+mx
            y_max=round(0.75*x_max,1) + my
            x_min=0.+mx
            y_min=0+my

            
            longdist=max(round(labeldist/2.,2),.01)
            shortdist=max(round(labeldist/10.,2),.01)
            
            self.labels_label_entry.delete(0,tk.END) 
            self.labels_label_entry.insert(0,f"{labeldist}")
            self.longticks_entry.delete(0,tk.END) 
            self.longticks_entry.insert(0,f"{longdist}")
            self.shortticks_entry.delete(0,tk.END) 
            self.shortticks_entry.insert(0,f"{shortdist}")
            



        self.kleinster_wert_entry.delete(0,tk.END) 
        self.kleinster_wert_entry.insert(0,f"{x_min} , {x_max}")
        self.ybounds_entry.delete(0,tk.END) 
        self.ybounds_entry.insert(0,f"{y_min} , {y_max}")


            
        
    
    
    def show_processing(self):
        text='PROCESSING'
        font = cv2.FONT_HERSHEY_DUPLEX
        #imgtext=self.first_frame
        textsize = cv2.getTextSize(text, font, 3, 2)[0]
        
        textX = int((self.first_frame.shape[1] - textsize[0]) / 2)
        textY = int((self.first_frame.shape[0] + textsize[1]) / 2 )
        
        #print('TEXT',textX,textY,text)
        
        cv2.putText(self.first_frame, "PROCESSING", (textX,textY),font, 3, (0, 0, 255), 2,cv2.LINE_AA)
        cv2.imshow("Video", self.first_frame)
        cv2.waitKey(1)

    
    def insert_scale(self):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
        global start_x,start_y,end_x,end_y,scale_length,scale_image,x_shift,y_shift,pos_x,pos_y,lnorm, lstart_x, lstart_y, scale_markup
        if self.first_frame is None:
            cv2.destroyAllWindows()
            messagebox.showerror("Fehler","Zuerst Video laden, KO-System positionieren und Maßstab festlegen!")
            return
        
        if self.start_point is None:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            messagebox.showerror("Fehler","Zuerst muss das Koordinatensystem positioniert werden!")
            self.show_first_frame()
            return
        
        if lnorm==-1:
            cv2.destroyAllWindows()
            messagebox.showerror("Fehler","Zuerst muss der Maßstab gewählt werden!")
            self.show_first_frame()
            return
        
        if self.first_frame is not None and self.start_point is not None and self.end_point is not None and lnorm!=-1:
            self.show_processing()
            
            
####### POLARKOORDINATEN
            if self.polar:  
                linewidth=self.lw_slider.get()
                gridwidth=self.gw_slider.get() 
                if self.first_circle:
                    rmax=np.round(self.radius_screen/lnorm*float(self.cal.get()),2)
                    self.kleinster_wert_entry.delete(0,tk.END)
                    self.kleinster_wert_entry.insert(0,f"0.,{rmax}")
                    self.first_circle=False

                grayscale=str(1.0-self.grayscale_slider.get())
                grayscale2=str(1.0-self.grayscale_slider2.get())
                
                linewidth=self.lw_slider.get()
                gridwidth=self.gw_slider.get()

                scale_length=self.radius_screen

                rax = self.kleinster_wert_entry.get()
                #xmax = float(self.groesster_wert_entry.get())
                rmin,rmax=map(float,rax.split(','))
                rmark=[]
                rmark.append(rmax)

                thetax = self.ybounds_entry.get()
                theta_min, theta_max = map(float,thetax.split(','))

                screenheight = root.winfo_screenheight()
                imageheight=self.first_frame.shape[0]

                px = 2*1/plt.rcParams['figure.dpi']  # pixel in inches
                known_length_in_pixels=lnorm
                known_length_in_m=float(self.cal.get())
                if screenheight<=imageheight:
                    scaling=screenheight/imageheight
                    pixels_x=5*(rmax-rmin)/known_length_in_m*known_length_in_pixels*screenheight/imageheight
                    
                else:
                    pixels_x=5*(rmax-rmin)/known_length_in_m*known_length_in_pixels
                    scaling=1.
                

               # fig, ax = plt.subplots(figsize=(pixels_x*px, pixels_x*px),layout='constrained',subplot_kw={'projection': 'polar'})
                #fig, ax = plt.subplots(figsize=(pixels_x*px, pixels_x*px),subplot_kw={'projection': 'polar'})
                fig, ax = plt.subplots(1,1,figsize=(pixels_x*px, pixels_x*px),subplot_kw={'projection': 'polar'})
                #fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
                #fig, ax = plt.subplots(1,1,figsize=(pixels_x*px, pixels_x*px),layout='constrained',subplot_kw={'projection': 'polar'})
                #fig.subplots_adjust(left=0.1,right=0.9)
               
               
                Transparency=not(self.bg.get())
                if not Transparency:     
                    fig.patch.set(alpha=0.0)
                    ax.patch.set(alpha=0.0)
                    #Hintergrundkreis zeichnen

                    # Create a dummy text to calculate the bounding box
                    text = ax.text(0.5, 0.5, 'XX360°XX', fontsize=1.5*2*int(self.lsize.get()), transform=ax.transAxes)
                    
                    # Draw the figure to calculate the size in pixels
                    #plt.draw()
                    fig.canvas.draw()
                    
                    # Get the bounding box in pixels
                    bbox = text.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
                    # Calculate the diagonal of the bounding box
                    diagonal_in_pixels = np.sqrt(bbox.width**2 + bbox.height**2)
                    
                    # Assuming the diagonal in pixels corresponds to a proportional delta_r in units
                    fig_width_in_pixels = ax.get_window_extent().width
                    delta_r = (diagonal_in_pixels / fig_width_in_pixels) * rmax
                    
                    
                    # Remove the dummy text
                    text.remove()


                    #rmax_coords = (0,rmax)  # Datenkoordinaten des Anheftpunkts
                    #r0_coords=(0,0)
                    #print("rmax",rmax)
                    #rmax_display_coords = ax.transData.transform(rmax_coords)  # Transformiere zu Display Koordinaten
                    #r0_display_coords = ax.transData.transform(r0_coords)  # Transformiere zu Display Koordinaten
                    #rmax_display_coords=(rmax_display_coords[0],rmax_display_coords[1])
                    #rdist=np.sqrt((rmax_display_coords[1]-r0_display_coords[1])**2.+(rmax_display_coords[0]-r0_display_coords[0])**2.)
                    #print("Display",rmax_display_coords,r0_display_coords,rdist)
                    #print("PIXELS",pixels_x)
                    #rmax_coords = ax.transData.inverted().transform(rmax_display_coords)
                    #print("rmax_coords",rmax_coords,2*int(self.lsize.get()),rmax)
                    #randnorm=rmax/rdist/pixels_x
                    #print("RANDNORM",randnorm)
                    #ax.add_patch(Circle((0,0),radius=1.7,transform=ax.transData._b,facecolor='white',clip_on=False))
                    #print("Labelsize",int(self.lsize.get())/pixels_x)
                    #rel_lsize=int(self.lsize.get())/pixels_x
                     

                    #ax.add_patch(Circle((0,0),radius=rmax+rel_lsize*11*randfaktor,transform=ax.transData._b,facecolor='white',clip_on=False))
                    #r_bounds = [0, rmax]
                    if self.polarscale_length.get():
                        tminbak=theta_min
                        tmaxbak=theta_max
                        theta_min=tminbak/2./np.pi/rmax*360.
                        theta_max=tmaxbak/2./np.pi/rmax*360.
                        if theta_max>=360. and theta_max<=363.: theta_max=360.
                        thetabounds= np.deg2rad(np.arange(theta_min, theta_max, 1./180.))
                        theta_min=tminbak
                        theta_max=tmaxbak
                    else:
                        thetabounds= np.deg2rad(np.arange(theta_min, theta_max, 1./180.))
                    #ax.fill_between(thetabounds,0,rmax+int(self.lsize.get())*7*known_length_in_m/known_length_in_pixels,color='white',clip_on=False)
                    #ax.fill_between(thetabounds,0,rmax+10.*int(self.lsize.get())*known_length_in_m/known_length_in_pixels/scaling,color='white',clip_on=False)
                    #ax.fill_between(thetabounds,0,rmax+50000*known_length_in_m/known_length_in_pixels/imageheight*2*int(self.lsize.get())/12,color='white',clip_on=False)
                    ax.fill_between(thetabounds,0,rmax+delta_r,color='white',clip_on=False)
                    #fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
                else:
                    fig.patch.set(alpha=0.0)
                    ax.patch.set(alpha=0.0)
                
                
                labeldist = float(self.labels_label_entry.get())
                shortdist = float(self.shortticks_entry.get())
                longdist = float(self.longticks_entry.get())

                if theta_max==360.:
                    ticks = np.arange(theta_min, theta_max+labeldist, labeldist)
                    #mticks sind die minor ticks
                    mticks = np.arange(theta_min, theta_max, shortdist)
                else:
                    ticks = np.arange(theta_min, theta_max+labeldist, labeldist)
                    #mticks sind die minor ticks
                    mticks = np.arange(theta_min, theta_max, shortdist)

                if self.polarscale_deg.get():
                    thetagrids=ticks.copy()
                    #thetalabels=np.round(ticks,0)
                    thetalabels=[]
                    for i in ticks:
                        thetalabels.append(str(int(round(i,0)))+"°")
                elif self.polarscale_length.get():
                    s_min,s_max=theta_min,theta_max
                    theta_min=s_min/2./np.pi/rmax*360.
                    theta_max=s_max/2./np.pi/rmax*360.
                    if theta_max>=360. and theta_max<=363.: theta_max=360.
                    labeldist_theta=labeldist/2./np.pi/rmax*360.    
                    ticks = np.arange(theta_min, theta_max+labeldist_theta, labeldist_theta)
                    #print(ticks,theta_max)
                    thetagrids=ticks.copy()
                    thetalabels=np.round(ticks*2*np.pi/360.*rmax,1)
                elif self.polarscale_rad.get():
                    thetagrids=ticks.copy()
                    thetalabels=[]
                    for i in ticks:
                        thetalabels.append(str(round(i/180.,2))+"π")

                if not self.show_xlabels.get():
                    thetagrids=ticks.copy()
                    thetalabels=[]
                    for i in ticks:
                        thetalabels.append(None)

                if self.invertvar_y.get():
                    ax.set_theta_direction(-1)
                    
                if not self.zerolabel.get() and not self.polarscale_length.get():
                    del thetalabels[0]
                    #print(thetagrids,thetalabels)
                    thetagrids=np.delete(thetagrids,[0])
                    
                if not self.lastlabelvar.get() and not self.polarscale_length.get():
                    del thetalabels[-1]
                    thetagrids=np.delete(thetagrids,[-1])

                
               
                
                ax.set_thetagrids(thetagrids,thetalabels,c=grayscale,fontsize=2*int(self.lsize.get()))
                
                

                ax.grid(visible=True, which='major', color=grayscale2, linewidth=gridwidth, linestyle='-', alpha=1.0)
                ax.grid(visible=True, which='minor', color=grayscale, linewidth=gridwidth, linestyle='-', alpha=1.0)

                ax.spines["polar"].set_color(grayscale)
                ax.spines[:].set_color(grayscale)
                ax.spines[:].set_linewidth(linewidth)

               
               

                ax.set_rmin(rmin)
                ax.set_rmax(rmax)
                ax.set_thetamin(theta_min)
                ax.set_thetamax(theta_max)


                ee_x,ee_y=self.end_point
                print("START",start_x,end_x,"END",ee_x,ee_y)
                if ee_x!=start_x:
                    theta_off=-round(np.arctan((ee_y-start_y)/(ee_x-start_x))*180./np.pi,0)
                elif ee_x==start_x and ee_y>start_y:
                    theta_off=90.
                else:
                    theta_off=-90.

                print("OFFSET",theta_off)
                #theta_off=-10.

                if theta_off<=0. and end_y<start_y:
                    theta_off=theta_off-180.
                if ee_x<start_x and ee_y>start_y:
                    theta_off=theta_off+180.
                print("OFFSET",theta_off)
                ax.set_theta_zero_location("E", offset=theta_off)
             

                grayscale=str(1.0-self.grayscale_slider.get())
                grayscale2=str(1.0-self.grayscale_slider2.get())
                
                linewidth=self.lw_slider.get()
                gridwidth=self.gw_slider.get()

                tickl1=self.ticklength.get()
                tickl2=tickl1/2

                ax.tick_params(axis='both', which='both', labelsize=2*int(self.lsize.get()), colors = grayscale,pad=2*int(self.lsize.get()))
                ax.tick_params(axis='y', which='both', labelsize=1.8*int(self.lsize.get()), colors = grayscale)
                
                # get all the labels of this axis
                labels = ax.get_yticklabels()
                # remove the first and the last labels
                labels[-1] = ""
                # set these new labels
                ax.set_yticklabels(labels)

                if self.show_xlabels.get():
                    tickfactor=np.round(1-float(self.ticklength.get())/20.*.06,2)
                    longtickfactor=np.round(1-float(self.ticklength.get())/20.*.12,2)
                    #print("TICKFAKTOR",tickfactor)
                    tick = [ax.get_rmax(), ax.get_rmax() * tickfactor]
                    longtick = [ax.get_rmax(), ax.get_rmax() * longtickfactor]  
                    # Iterate the points between 0 to 360 with step=10
                    if not self.polarscale_length.get():
                        for t in np.deg2rad(np.arange(theta_min, theta_max, shortdist)):
                            ax.plot([t, t], tick, lw=gridwidth, color = grayscale)
                        for t in np.deg2rad(np.arange(theta_min, theta_max, longdist)):
                            ax.plot([t, t], longtick, lw=gridwidth, color = grayscale) 
                    else:
                        for t in np.deg2rad(np.arange(theta_min, theta_max, shortdist/2./np.pi/rmax*360.)):
                            ax.plot([t, t], tick, lw=gridwidth, color = grayscale)
                        for t in np.deg2rad(np.arange(theta_min, theta_max, longdist/2./np.pi/rmax*360.)):
                            ax.plot([t, t], longtick, lw=gridwidth, color = grayscale)
                        
                    
                    

                if(not self.show_ylabels.get()):
                    ax.grid(axis='both',visible=False)
                    labels=[]
                    ax.set_yticklabels(labels)
                else:
                    ax.grid(axis='y',which='both',visible=True)
                    
                    if not self.show_xlabels.get():
                        ax.grid(axis='y',visible=True)
                        #ax.spines['theta'].set_visible=False
                        ax.set_thetagrids([])
                    
                #plt.scatter([0],[0], c=grayscale, marker="+",clip_on=False)
                ax.plot([0],[0],marker="+",markersize=20,color=grayscale,clip_on=False)


               
                
                ax.set_rmin(rmin)
                ax.set_rmax(rmax)
                ax.set_thetamin(theta_min)
                ax.set_thetamax(theta_max)
                
                
                #fig.set_size_inches(1.5*pixels_x*px,1.5*pixels_x*px)
                #fig.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)  
              
                
                #fig.tight_layout()

                fig.canvas.draw()
                #fig.tight_layout()
                fig.subplots_adjust(left=0.25, right=0.75, top=0.75, bottom=0.25)  
                fig.canvas.draw()

                imbuf = fig.canvas.buffer_rgba()
                imbufnp = np.asarray(imbuf)
                scale_image = cv2.cvtColor(imbufnp, cv2.COLOR_RGBA2BGRA)
                #cv2.imshow("Test",scale_image)
                
                ############## PIXELKOORDINATEN VON URSPRUNG UND ANHEFTPUNKT BESTIMMEN
                
                # Berechne die Pixelkoordinaten des Ursprungs
                origin_data_coords = (0, 0)  # Datenkoordinaten des Anheftpunkts
                origin_display_coords = ax.transData.transform(origin_data_coords)  # Transformiere zu Display Koordinaten

                
            
                # Verwende fig.dpi_scale_trans, um Display-Koordinaten in Pixelkoordinaten umzurechnen
                dpi=plt.rcParams['figure.dpi']
                origin_pixel_coords = fig.dpi_scale_trans.inverted().transform(origin_display_coords)
                origin_pixel_coords = origin_pixel_coords*dpi
                print('ORIGIN KO',origin_pixel_coords)
                
                 
                marker_coords = (0,rmax)
                marker_display_coords = ax.transData.transform(marker_coords)  # Transformiere zu Display Koordinaten
                marker_pixel_coords = fig.dpi_scale_trans.inverted().transform(marker_display_coords)
                marker_pixel_coords = marker_pixel_coords*dpi
                
                

                ### EXPORTIERBARES BILD ERZEUGEN######################################################################################
                grayscale="0."
                grayscale2="0."
                ax.grid(visible=True, which='major', color=grayscale2, linewidth=gridwidth, linestyle='-', alpha=1.0)
                ax.grid(visible=True, which='minor', color=grayscale, linewidth=gridwidth, linestyle='-', alpha=1.0)

                ax.spines["polar"].set_color(grayscale)
                ax.spines[:].set_color(grayscale)
                ax.spines[:].set_linewidth(linewidth)

                Transparency=not(self.bg.get())
                
                fig.patch.set(alpha=1.0)
                ax.patch.set(alpha=1.0)

                
                linewidth=self.lw_slider.get()
                gridwidth=self.gw_slider.get()

                tickl1=self.ticklength.get()
                tickl2=tickl1/2

                ax.tick_params(axis='both', which='both', labelsize=2*int(self.lsize.get()), colors = grayscale,pad=2*int(self.lsize.get()))
                ax.tick_params(axis='y', which='both', labelsize=1.8*int(self.lsize.get()), colors = grayscale)
                     

                if(not self.show_ylabels.get()):
                    ax.grid(axis='both',visible=False)
                    labels=[]
                    ax.set_yticklabels(labels)
                else:
                    ax.grid(axis='y',visible=True)

                ax.plot([0],[0],marker="+",markersize=20,color='black',clip_on=False)
                ax.set_rmin(rmin)
                ax.set_rmax(rmax)
                ax.set_thetamin(theta_min)
                ax.set_thetamax(theta_max)


                fig.canvas.draw()   

                imbuf = fig.canvas.buffer_rgba()
                imbufnp = np.asarray(imbuf)
                self.export_axis = cv2.cvtColor(imbufnp, cv2.COLOR_RGBA2BGRA)
                 
               
                
                red_pixel_coords=(scale_image.shape[0]-origin_pixel_coords[1],origin_pixel_coords[0])

                
                print("Pixelkoordinaten des Anheftpunkts:", red_pixel_coords)
                
                blue_pixel_coords=(scale_image.shape[0]-marker_pixel_coords[1],marker_pixel_coords[0])
                
                self.center_coords=np.array([[origin_pixel_coords[0]],[scale_image.shape[0]-origin_pixel_coords[1]],[1]])
                

            
                print("Pixelkoordinaten des Markierungspunkts:", blue_pixel_coords) 
                y1,x1=red_pixel_coords
                y2,x2=blue_pixel_coords
                dx=x2-x1
                dy=y2-y1

                ds=np.sqrt(dx*dx+dy*dy)
                
                
                markerdist_in_m=rmax
                markerdist_in_pixels_as_is=ds
                known_length_in_pixels=lnorm
                known_length_in_m=float(self.cal.get())
                pixels_per_m=known_length_in_pixels/known_length_in_m
                
                if self.units_per_px is not None:
                    pixels_per_m=1./self.units_per_px
                required_markerdist_in_pixels=pixels_per_m*markerdist_in_m

                scaling_factor=required_markerdist_in_pixels/markerdist_in_pixels_as_is
                self.scaling_factor=scaling_factor
                print('SCALING FACTOR',scaling_factor)
                
 
                self.scale_image_not_rot=scale_image
                

                end_x_temp=end_x
                end_y_temp=end_y
                end_x=start_x+self.radius_screen
                end_y=start_y

                #IMAGE DER SKALA DREHEN
                self.make_rotated_image()
                end_x=end_x_temp
                end_y=end_y_temp


                #IMAGE DER SKALA EINFÜGEN
                self.overlay_image(self.first_frame)
                scale_added=True

                #Ersten frame mit Skala anzeigen
                #cv2.imshow("Video mit Skala", self.first_frame)
                self.show_first_frame()


                
########### KARTHESISCHES KO-SYSTEM /SKALA

            if not self.polar:
                #start_x, start_y = self.start_point
                #end_x, end_y = self.end_point
                grayscale=str(1.0-self.grayscale_slider.get())
                grayscale2=str(1.0-self.grayscale_slider2.get())
                
                linewidth=self.lw_slider.get()
                gridwidth=self.gw_slider.get()
                
                delta_x = end_x - start_x
                delta_y = end_y - start_y
                
                #Länge des Vergleichsmaßstabs in Pixel (=lnorm(?))
                scale_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
                
                
                # Erstelle ein hochaufgelöstes Bild mit Skala mithilfe von matplotlib
                
                xax = self.kleinster_wert_entry.get()
                #xmax = float(self.groesster_wert_entry.get())
                xmin,xmax=map(float,xax.split(','))

                yax = self.ybounds_entry.get()
                ymin, ymax = map(float,yax.split(',')) 
                #ticks_frequency = 0.1

                
                #DPI = ((Vergleichsmassstabslänge in px/Vergleichsmassstabslänge in Skaleneinheiten)*(xmin-xmax))/10 (durch 10, weil die Figur 10 inches breit ist)
                #dpi_im = lnorm/float(self.cal.get())*(xmax-xmin)/10.*1.5
                width_px = root.winfo_screenwidth()
                screenheight = root.winfo_screenheight()
                imageheight=self.first_frame.shape[0]

                dpi=plt.rcParams['figure.dpi']
                px = 1/plt.rcParams['figure.dpi']  # pixel in inches
                known_length_in_pixels=lnorm
                known_length_in_m=float(self.cal.get())
                if screenheight<=imageheight:
                    pixels_x=4*max((xmax-xmin),(ymax-ymin))/known_length_in_m*known_length_in_pixels*screenheight/imageheight 
                    #pixels_x=2*max((xmax-xmin),(ymax-ymin))/known_length_in_m*known_length_in_pixels 
                    #pixels_x=int((screenheight+imageheight)/2)
                     
                else:
                    pixels_x=2*max((xmax-xmin),(ymax-ymin))/known_length_in_m*known_length_in_pixels  
                
               
                print('PIXELS_X',pixels_x*px,"px=",px,"1/px=",1/px)
               
                fig, ax = plt.subplots(figsize=(pixels_x*px, pixels_x*px),layout='constrained')


                Transparency=not(self.bg.get())
                if not Transparency:     
                    fig.patch.set(alpha=0.0)
                    ax.patch.set(alpha=0.0)
                    #Hintergrundkreis zeichnen

                    # Create a dummy text to calculate the bounding box
                    text = ax.text(0.5, 0.5, 'XXXX', fontsize=1.*2*int(self.lsize.get()), transform=ax.transAxes)
                    
                    # Draw the figure to calculate the size in pixels
                    #plt.draw()
                    fig.canvas.draw()
                    
                    # Get the bounding box in pixels
                    bbox = text.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
                    # Calculate the diagonal of the bounding box
                    diagonal_in_pixels = np.sqrt(bbox.width**2 + bbox.height**2)
                    
                    # Assuming the diagonal in pixels corresponds to a proportional delta_r in units
                    fig_width_in_pixels = ax.get_window_extent().width
                    delta_r = (diagonal_in_pixels / fig_width_in_pixels) * (xmax-xmin)
                    
                    
                    # Remove the dummy text
                    text.remove()
               
                
                shortdist = float(self.shortticks_entry.get())
                longdist = float(self.longticks_entry.get())
                
                if self.kosys.get():
                    ax.set(xlim=(xmin-shortdist, xmax+longdist))
                else:
                    ax.set(xlim=(xmin-0.5*longdist, xmax+longdist))
                    
                ticks = np.arange(xmin, xmax+longdist, longdist)
                #mticks sind die minor ticks
                mticks = np.arange(xmin, xmax, shortdist)
                labeldist=float(self.labels_label_entry.get())
                
                #so oft passt labldist in longdist:
                n=int(labeldist/longdist)
                if self.show_xlabels.get():
                    labels = [round(t, 1) if i % n == 0 else None for i, t in enumerate(ticks)]
                else:
                    labels = [None for i, t in enumerate(ticks)]

                
                for i,t in enumerate(ticks):
                    if labels[i]==-0.0 or labels[i]==0.0:
                        if not self.zerolabel.get():
                            labels[i]=None
                        else:
                            labels[i]=0

                #print(labels)
                
                
                ax.set_xticks(ticks, minor=False)
                ax.set_xticks(mticks, minor=True)
                ax.set_xticklabels(labels)

                if self.kosys.get():
                    #Y-ACHSE: Achsenverhältnis auf 1
                    ax.set(ylim=(ymin-shortdist, ymax+longdist),aspect=1)
                    #ax.set(ylim=(ymin, ymax+longdist),aspect=1)
                    yticks = np.arange(ymin, ymax+longdist, longdist)
                    #mticks sind die minor ticks
                    ymticks = np.arange(ymin, ymax, shortdist)
                
                    if self.show_ylabels.get():
                        ylabels = [round(t, 1) if i % n == 0 else None for i, t in enumerate(yticks)]
                    else:
                        ylabels = [ None for i, t in enumerate(yticks)]

                    for i,t in enumerate(yticks):
                        if ylabels[i]==-0.0 or ylabels[i]==0.0: ylabels[i]=None

                    #print(labels)
                    
                    
                    ax.set_yticks(yticks, minor=False)
                    ax.set_yticks(ymticks, minor=True)
                    ax.set_yticklabels(ylabels)
                else:
                    ax.set_yticks([])
                    ax.set(ylim=(ymin-3*longdist, ymax+longdist),aspect=1)
                    
                

                # Set bottom and left spines as x and y axes of coordinate system
                ax.spines['bottom'].set_position('zero')
                ax.spines['left'].set_position('zero')
                ax.spines['bottom'].set_linewidth(linewidth)
                ax.spines['left'].set_linewidth(linewidth)
                
            
                
                ax.spines['bottom'].set_color(grayscale)
                ax.spines['left'].set_color(grayscale)

                # Remove top and right spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                if not self.kosys.get():
                    ax.spines['left'].set_visible(False)
                else:
                    ax.spines['left'].set_visible(True)
              


                # Draw major and minor grid lines
                ax.grid(which='major', color=grayscale2, linewidth=gridwidth, linestyle='-', alpha=1.0)
                
                tickl1=self.ticklength.get()
                tickl2=tickl1/2
                
                
                inverty=self.invertvar_y.get()
                invertx=self.invertx
                
                
                ax.tick_params(axis='both', which='major', labelsize=2*int(self.lsize.get()), width=gridwidth, length=tickl1, colors = grayscale)
                ax.tick_params(axis='both', which='minor', labelsize=2*int(self.lsize.get()), width=gridwidth, length=tickl2, colors = grayscale)
                
                
                if inverty:
                    ax.tick_params(axis="x",which='both', direction="in", pad=-2*int(1.5*float(self.lsize.get())))
                else:
                    ax.tick_params(axis="x",which='both', direction="out", pad=2*int(.3*float(self.lsize.get())))
                
                ax.spines['bottom'].set_color(grayscale)
                ax.spines['left'].set_color(grayscale)
                
                if self.kosys.get():
                    ax.grid(visible=True, which='major', color=grayscale2, linewidth=gridwidth, linestyle='-', alpha=1.0)
                else:
                    ax.grid(visible=False)
                
                arrow_fmt = dict(markersize=int(2*int(self.lsize.get())/2), color=grayscale, clip_on=False)
                if not invertx:
                    #arrow_fmt = dict(markersize=6, color='black', clip_on=False)
                    ax.plot((1), (0), marker='>', transform=ax.get_yaxis_transform(), **arrow_fmt)
            
                    
                else:
                    ax.invert_xaxis()
                    #arrow_fmt = dict(markersize=6, color='black', clip_on=False)
                    ax.plot((0), (0), marker='<', transform=ax.get_yaxis_transform(), **arrow_fmt)
                
                if self.kosys.get():
                    if not inverty:
                        #arrow_fmt = dict(markersize=6, color='black', clip_on=False)
                        ax.plot((0), (1), marker='^', transform=ax.get_xaxis_transform(), **arrow_fmt)
                    else:
                        ax.invert_yaxis()
                        #arrow_fmt = dict(markersize=6, color='black', clip_on=False)
                        ax.plot((0), (0), marker='v', transform=ax.get_xaxis_transform(), **arrow_fmt)    

                        
                #Transparenz setzen bzw. Hintergrundrechteck hinter Skala zeichnen
                        

                Transparency=not(self.bg.get())
                if Transparency:     
                    fig.patch.set(alpha=0.0)
                    ax.patch.set(alpha=0.0)
                else:
                    if not self.kosys.get():
                        fig.patch.set(alpha=0.0)
                        ax.patch.set(alpha=0.0)
                        # Berechne die Pixelkoordinaten des Ursprungs
                        #lower_limit_coords = (xmin, 0)  # Datenkoordinaten des Anheftpunkts
                        #lower_limit_display_coords = ax.transData.transform(lower_limit_coords)  # Transformiere zu Display Koordinaten
                        #lower_limit_display_coords=(lower_limit_display_coords[0],lower_limit_display_coords[1]-4.5*int(self.lsize.get()))
                        #lower_limit_coords = ax.transData.inverted().transform(lower_limit_display_coords)
                        #print("Lower Limit Coords",lower_limit_coords,2*int(self.lsize.get()))
                        #ax.plot(lower_limit_coords[0],lower_limit_coords[1],marker='o')


                        
                        #ax.add_patch(Rectangle((xmin-longdist,lower_limit_coords[1]), (xmax-xmin)+3*longdist,abs(lower_limit_coords[1])*1.2 ,facecolor='white',edgecolor='white',clip_on=False))
                        ax.add_patch(Rectangle((xmin-shortdist-delta_r,ymin-delta_r), (xmax-xmin)+2.*delta_r+longdist,1.3*delta_r ,facecolor='white',edgecolor='white',clip_on=False))

                    else:
                        fig.patch.set(alpha=0.0)
                        ax.patch.set(alpha=0.0)
                         # Berechne die Pixelkoordinaten des Ursprungs
                        #lower_limit_coords = (xmin, 0)  # Datenkoordinaten des Anheftpunkts
                        #lower_limit_display_coords = ax.transData.transform(lower_limit_coords)  # Transformiere zu Display Koordinaten
                        #lower_limit_display_coords=(lower_limit_display_coords[0],lower_limit_display_coords[1]-4.5*int(self.lsize.get()))
                        #lower_limit_coords = ax.transData.inverted().transform(lower_limit_display_coords)
                        #print("Lower Limit Coords",lower_limit_coords,2*int(self.lsize.get()))
                        #ax.plot(lower_limit_coords[0],lower_limit_coords[1],marker='o')
                        ax.add_patch(Rectangle((xmin-shortdist-delta_r,ymin-shortdist-delta_r), (xmax-xmin)+2.*delta_r+longdist,2.*delta_r+ymax-ymin+longdist ,facecolor='white',edgecolor='white',clip_on=False))



                
                

                #fig.tight_layout()
                #fig.canvas.draw()
                #fig.tight_layout()
                #fig.canvas.draw()
                #fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)  
                fig.canvas.draw()
                
                # Koordinaten des Anheftpunkts auslesen
                markstr=self.lower_marker.get()
                mx,my = map(float,markstr.split(',')) 
            
            
                # Berechne die Pixelkoordinaten des Ursprungs
                origin_data_coords = (mx, my)  # Datenkoordinaten des Anheftpunkts
                origin_display_coords = ax.transData.transform(origin_data_coords)  # Transformiere zu Display Koordinaten

                
            
                # Verwende fig.dpi_scale_trans, um Display-Koordinaten in Pixelkoordinaten umzurechnen
                origin_pixel_coords = fig.dpi_scale_trans.inverted().transform(origin_display_coords)
                origin_pixel_coords = origin_pixel_coords*dpi
                print('ORIGIN KO',origin_pixel_coords)
                
                 
                marker_coords = (xmax,0)
                marker_display_coords = ax.transData.transform(marker_coords)  # Transformiere zu Display Koordinaten
                marker_pixel_coords = fig.dpi_scale_trans.inverted().transform(marker_display_coords)
                marker_pixel_coords = marker_pixel_coords*dpi
                


                imbuf = fig.canvas.buffer_rgba()
                imbufnp = np.asarray(imbuf)
                scale_image = cv2.cvtColor(imbufnp, cv2.COLOR_RGBA2BGRA)     
                
                
                #EXPORTIERBARES BILD ERZEUGEN
                ax.tick_params(axis='both', which='major', labelsize=2*int(self.lsize.get()), width=gridwidth, length=tickl1, colors = 'black')
                ax.tick_params(axis='both', which='minor', labelsize=2*int(self.lsize.get()), width=gridwidth, length=tickl2, colors = 'black')
                ax.spines['bottom'].set_color('black')
                ax.spines['left'].set_color('black')
                ax.grid(which='major', color='black', linewidth=gridwidth, linestyle='-', alpha=1.0)

                if self.kosys.get():
                    ax.grid(visible=True, which='major', color=grayscale2, linewidth=gridwidth, linestyle='-', alpha=1.0)
                else:
                    ax.grid(visible=False)
                
                
                arrow_fmt = dict(markersize=int(2*int(self.lsize.get())/2), color='black', clip_on=False)
                if not invertx:
                    ax.plot((1), (0), marker='>', transform=ax.get_yaxis_transform(), **arrow_fmt)
                else:
                    ax.plot((0), (0), marker='<', transform=ax.get_yaxis_transform(), **arrow_fmt)
                
                if self.kosys.get():
                    if not inverty:
                        ax.plot((0), (1), marker='^', transform=ax.get_xaxis_transform(), **arrow_fmt)
                    else:
                        ax.plot((0), (0), marker='v', transform=ax.get_xaxis_transform(), **arrow_fmt)
                    
                #fig.savefig(os.path.join(self.basedir,"KO-sys Export Version.png"), bbox_inches='tight', pad_inches=0, transparent=False,dpi=200)
                fig.patch.set(alpha=1.0)
                ax.patch.set(alpha=1.0)


                fig.canvas.draw()   

                imbuf = fig.canvas.buffer_rgba()
                imbufnp = np.asarray(imbuf)
                self.export_axis = cv2.cvtColor(imbufnp, cv2.COLOR_RGBA2BGRA)   

                
                         
                red_pixel_coords=(scale_image.shape[0]-origin_pixel_coords[1],origin_pixel_coords[0])
                
                print("Pixelkoordinaten des roten Punkts:", red_pixel_coords)
                
                blue_pixel_coords=(scale_image.shape[0]-marker_pixel_coords[1],marker_pixel_coords[0])


            
                print("Pixelkoordinaten des blauen Punkts:", blue_pixel_coords,marker_pixel_coords) 
                y1,x1=red_pixel_coords
                y2,x2=blue_pixel_coords
                dx=x2-x1
                dy=y2-y1

                ds=np.sqrt(dx*dx+dy*dy)
                
                #self.center_coords=[[round(trimmed_axis_image.shape[0]-origin_pixel_coords[1])],[round(origin_pixel_coords[0])],[1]]
                
                self.center_coords=np.array([[origin_pixel_coords[0]],[scale_image.shape[0]-origin_pixel_coords[1]],[1]])
                
                
                
                self.markerdist_in_m=xmax-mx
                print("MARKERDIST",self.markerdist_in_m)
                self.markerdist_in_pixels_as_is=ds
                print("MARKERDIST IN PIX",self.markerdist_in_pixels_as_is)
                known_length_in_pixels=lnorm
                print("KNOWN LENGTH IN PIX",lnorm)
                known_length_in_m=float(self.cal.get())
                print("KNOWN LENGTH IN M",known_length_in_m) 
                pixels_per_m=known_length_in_pixels/known_length_in_m
                #print("UNITS PER PIXELS BEFORE:",1./pixels_per_m)
                
                #self.units_per_px=1./pixels_per_m
                #required_markerdist_in_pixels=pixels_per_m*markerdist_in_m

                #scaling_factor=required_markerdist_in_pixels/markerdist_in_pixels_as_is
                
                scaling_factor=known_length_in_pixels/self.markerdist_in_pixels_as_is*self.markerdist_in_m/known_length_in_m
                if self.units_per_px is not None:
                    print("NEU SKALIERT!")
                    scaling_factor=1./self.units_per_px*self.markerdist_in_m/self.markerdist_in_pixels_as_is
                
                
                print('SCALING FACTOR',scaling_factor)
                self.scaling_factor=scaling_factor
                 
                
                
                self.scale_image_not_rot=scale_image.copy()
                #cv2.imshow("Test",self.scale_image_not_rot)
                
                #IMAGE DER SKALA DREHEN
                self.make_rotated_image()

                
                print("RED PX KO BERECHNET!!", red_pixel_coords)
                print("SHIFT",y_shift,x_shift)
                #y_shift,x_shift=red_pixel_coords
                #self.y_shift,self.x_shift=red_pixel_coords

                #x1=scale_image.shape[1]-x1
                print("VERSCHIEBUNG DES EINGEFÜGTEN BILDES",y_shift,x_shift)
                #print("Pixelkoordinaten des blauen Punkts:", blue_pixel_coords) 
                print(scale_image.shape)

                #IMAGE DER SKALA EINFÜGEN
                self.overlay_image(self.first_frame)
                scale_added=True

                #Ersten frame mit Skala anzeigen
                #cv2.imshow("Video mit Skala", self.first_frame)
                self.show_first_frame()


    def make_rotated_image(self):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
        global start_x,start_y,end_x,end_y,scale_length,scale_image,x_shift,y_shift,pos_x,pos_y,lnorm, lstart_x, lstart_y, scale_markup


        angle = np.arctan2(end_y - start_y, end_x - start_x) * (180 / np.pi)
        if self.invertx:
            angle=angle-180.
        scale_image = self.rotate_image(angle)
        #scale_markup=self.rotate_image_2(angle)
        print('Dimensionen des gedrehten Bildes')
        print(scale_image.shape)


        # Finde die Koordinaten der markierten Pixel im gedrehten Bild
        print('Übergebene Bildabmessungen',scale_image.shape)
       
        #cv2.imwrite("gedreht.png", scale_image)       
        

        #shift_y = scale_image.shape[0]
        # Lege das skalierte Bild über den ersten Frame

    def overlay_image(self, background):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
        global start_x,start_y,end_x,end_y,scale_length,scale_image,x_shift,y_shift,pos_x,pos_y,lnorm, lstart_x, lstart_y
        #print('x-shift,y-shift',x_shift,y_shift)
        overlay=scale_image
        ys_min=0
        ys_max=overlay.shape[0]
        xs_min=0
        xs_max=overlay.shape[1]
        
        alpha=1.-float(self.alpha_slider.get())
        
        alpha_s = overlay[:, :, 3] / 255.0*alpha
        #alpha_s = 0.5
        alpha_l = 1.0 - alpha_s
        pos_x=start_x
        pos_y=start_y
        y1, y2 = pos_y-y_shift, pos_y-y_shift + overlay.shape[0]
        x1, x2 = pos_x-x_shift, pos_x-x_shift + overlay.shape[1]

        if y1<0:
            ys_min=abs(y1)
            y1=0
        if x1<0:
            xs_min=abs(x1)
            x1=0
        if y2>background.shape[0]:
            ys_max=ys_max-(y2-background.shape[0])
            y2=background.shape[0]
        if x2>background.shape[1]:
            xs_max=xs_max-(x2-background.shape[1])
            x2=background.shape[1]

        for c in range(0, 3):
            background[y1:y2, x1:x2, c] = (alpha_s[ys_min:ys_max,xs_min:xs_max] * overlay[ys_min:ys_max, xs_min:xs_max, c] +
                                           alpha_l[ys_min:ys_max,xs_min:xs_max] * background[y1:y2, x1:x2, c])


        #cv2.imshow("Video mit Skala", background)


    def find_marker_pixel_coords(self, image, marker_bgr_up, marker_bgr_low):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
        global start_x,start_y,end_x,end_y,scale_length,scale_image,x_shift,y_shift,pos_x,pos_y, lnorm, lstart_x, lstart_y, scale_markup
        # Extrahiere die Koordinaten des markierten Pixels im Bild
        print('Farbcode:',marker_bgr_up)
        
        print('Bildabmessungen in find marker',image.shape)
        #mask = np.all(image[:, :, :3] >= comp, axis=-1)
        #print(mask)
        mask = np.logical_and(np.all(marker_bgr_up >= image[:, :, :3],axis=-1) , np.all(image[:, :, :3] >= marker_bgr_low, axis=-1))
        #mask = np.all(marker_bgr_up >= image[:, :, :3],axis=-1)

        
        #print(mask)
        coords = np.argwhere(mask)
        #print(coords)
        #print(coords.sum())

        # Mittelpunkt der markierten Pixel berechnen
        mean_coords = np.mean(coords, axis=0)
        print("Imageshape",image.shape)
        
        #mean_coords[0]=int(image.shape[0]-mean_coords[0])
        mean_coords[0]=int(mean_coords[0]) 
        #mean_coords[1]=int(image.shape[1]-mean_coords[1])
        mean_coords[1]=int(mean_coords[1])
        #print('y1,x1',mean_coords[0],mean_coords[1])                

        return tuple(mean_coords.astype(int))

#    def rotate_image(self, image, angle):
#        center = tuple(np.array(image.shape[1::-1]) / 2)
#        rot_mat = cv2.getRotationMatrix2D(center, -1.0*angle, 1.0)
#        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
#        return result

    def rotate_image(self, angle):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
        global start_x,start_y,end_x,end_y,scale_length,scale_image,x_shift,y_shift,pos_x,pos_y,lnorm, lstart_x, lstart_y, scale_markup
        # Rotiere das Bild um den gegebenen Winkel und schreibe es in ein rechteckiges Array ein
        image=scale_image
        self.rot_angle_bak=angle
        center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, -1.0*angle, self.scaling_factor)
        
        # Berechne die neuen Dimensionen des rechteckigen Arrays
        new_w = int(image.shape[1] * np.abs(np.cos(np.radians(angle))) +
                    image.shape[0] * np.abs(np.sin(np.radians(angle))))
        new_h = int(image.shape[0] * np.abs(np.cos(np.radians(angle))) +
                    image.shape[1] * np.abs(np.sin(np.radians(angle))))
        
        
        
        # Berechne die Verschiebung, um das rotierte Bild in das rechteckige Array einzuschreiben
        tx = (new_w - image.shape[1]) / 2
        ty = (new_h - image.shape[0]) / 2
        
        print(rot_mat)
        
        rot_mat[0, 2] += tx
        rot_mat[1, 2] += ty
        
        print(rot_mat)
        
        
        x_shift=int(round(np.dot(rot_mat,self.center_coords)[0,0],0))
        y_shift=int(round(np.dot(rot_mat,self.center_coords)[1,0],0))
            
        
        
        result = cv2.warpAffine(image, rot_mat, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255, 0))
        cv2.imwrite("gedreht.png",result)
        return result
    
    def rotate_image_2(self, angle):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
        global start_x,start_y,end_x,end_y,scale_length,scale_image,x_shift,y_shift,pos_x,pos_y,lnorm, lstart_x, lstart_y, scale_markup
        # Rotiere das Bild um den gegebenen Winkel und schreibe es in ein rechteckiges Array ein
        image=scale_markup
        center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, -1.0*angle, 1.0)
        
        # Berechne die neuen Dimensionen des rechteckigen Arrays
        new_w = int(image.shape[1] * np.abs(np.cos(np.radians(angle))) +
                    image.shape[0] * np.abs(np.sin(np.radians(angle))))
        new_h = int(image.shape[0] * np.abs(np.cos(np.radians(angle))) +
                    image.shape[1] * np.abs(np.sin(np.radians(angle))))
        
        # Berechne die Verschiebung, um das rotierte Bild in das rechteckige Array einzuschreiben
        tx = (new_w - image.shape[1]) / 2
        ty = (new_h - image.shape[0]) / 2
        
        rot_mat[0, 2] += tx
        rot_mat[1, 2] += ty
        
        result = cv2.warpAffine(image, rot_mat, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255, 0))
        
        return result
    
    def save_video_with_scale(self):
        global cached_frames, total_frames,current_frame,fps,fps_native,t0_frame,t0_time,start_point,scale_added, timestamp_added
        global start_x,start_y,end_x,end_y,scale_length,scale_image,x_shift,y_shift,pos_x,pos_y,lnorm, lstart_x, lstart_y
        global imageflag
        if self.first_frame is not None:
            cv2.destroyWindow("Video")
            if not imageflag:
                self.step=int(self.step_entry.get())
                self.output_filename = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")])
                if self.output_filename:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    if self.start_end_selected_crop:
                        video_out = cv2.VideoWriter(self.output_filename, fourcc, fps/self.step, (self.first_frame[self.crop_y0:self.crop_y1, self.crop_x0:self.crop_x1, :].shape[1], self.first_frame[self.crop_y0:self.crop_y1, self.crop_x0:self.crop_x1, :].shape[0]))
                    else:
                        video_out = cv2.VideoWriter(self.output_filename, fourcc, fps/self.step, (self.first_frame.shape[1], self.first_frame.shape[0]))
                    self.current_frame = self.frame_min
                    #self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                    #i=self.current_frame
                    for i in range(self.frame_min,self.frame_max+1,self.step):
                    
                        if i in cached_frames:    
                            frame_with_scale = cached_frames[i].copy()
                            
                            if scale_added:
                                self.overlay_image(frame_with_scale)

                            if timestamp_added:
                                current_frame=i
                                frame1=self.draw_timestamp(frame_with_scale)
                                frame_with_scale=frame1.copy()           
                        
                            if self.start_end_selected_crop:
                                frame_with_scale = frame_with_scale[self.crop_y0:self.crop_y1, self.crop_x0:self.crop_x1, :]
                            video_out.write(frame_with_scale)

                            cv2.imshow("Bearbeitetes Video", frame_with_scale)
                            cv2.waitKey(1)
                            #i=i+step
                            #self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)

                            #if i>=self.frame_max:
                            #    break

                            #for k in range (0,step-1):
                            #   ret, frame = self.cap.read()
                            #  if not ret:
                            #     break

                            #if cv2.waitKey(30) & 0xFF == 27:
                            #    break

                    video_out.release()
                    cv2.destroyWindow("Bearbeitetes Video")
                    messagebox.showinfo("Erfolgreich gespeichert", "Das Video wurde erfolgreich gespeichert.")
                    self.set_frame(0)
                    self.show_first_frame()
                    #x,y,w,h=cv2.getWindowImageRect("Video Editor")
                    x=self.root.winfo_x()
                    y=self.root.winfo_y()
                    w=self.root.winfo_width()
                    cv2.moveWindow("Video",x+w+30,y)
                else:
                    self.set_frame(0)
                    self.show_first_frame()
                    #x,y,w,h=cv2.getWindowImageRect("Video Editor")
                    x=self.root.winfo_x()
                    y=self.root.winfo_y()
                    w=self.root.winfo_width()
                    cv2.moveWindow("Video",x+w+30,y)
            else:
                self.output_filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("png files", "*.png")])
                if self.output_filename:
                    frame_with_scale=cached_frames[0].copy()
                    if scale_added:
                            self.overlay_image(frame_with_scale)           
                
                    if self.start_end_selected_crop:
                        frame_with_scale = frame_with_scale[self.crop_y0:self.crop_y1, self.crop_x0:self.crop_x1, :]
                    cv2.imwrite(self.output_filename,frame_with_scale)
                    messagebox.showinfo("Erfolgreich gespeichert", "Das Bild wurde erfolgreich gespeichert.")
                    self.set_frame(0)
                    self.show_first_frame()
                    #x,y,w,h=cv2.getWindowImageRect("Video Editor")
                    x=self.root.winfo_x()
                    y=self.root.winfo_y()
                    w=self.root.winfo_width()
                    cv2.moveWindow("Video",x+w+30,y)
                
    def save_scale(self):
        global start_x,start_y,end_x,end_y,scale_length,scale_image,x_shift,y_shift,pos_x,pos_y,lnorm, lstart_x, lstart_y
        if self.export_axis is not None:
            if self.first_frame is not None and scale_image is not None:
                cv2.destroyWindow("Video")
                self.output_filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("png files", "*.png")])
                if self.output_filename:
                    #axis_image=cv2.imread(os.path.join(self.basedir,"KO-sys Export Version.png"), cv2.IMREAD_UNCHANGED)
                    cv2.imwrite(self.output_filename,self.export_axis)
                    messagebox.showinfo("Erfolgreich gespeichert", "Das Bild wurde erfolgreich gespeichert.")
                    self.set_frame(0)
                    self.show_first_frame()
                    #x,y,w,h=cv2.getWindowImageRect("Video Editor")
                    x=self.root.winfo_x()
                    y=self.root.winfo_y()
                    w=self.root.winfo_width()
                    cv2.moveWindow("Video",x+w+30,y)
                else:
                    self.set_frame(0)
                    self.show_first_frame()
                    #x,y,w,h=cv2.getWindowImageRect("Video Editor")
                    x=self.root.winfo_x()
                    y=self.root.winfo_y()
                    w=self.root.winfo_width()
                    cv2.moveWindow("Video",x+w+30,y)



    def play_video_with_scale(self):
        if self.first_frame is not None:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Füge die Skala zu jedem Frame hinzu
                frame_with_scale = frame.copy()
                start_x, start_y = self.start_point
                end_x, end_y = self.end_point

                line_thickness = 2
                delta_x = end_x - start_x
                delta_y = end_y - start_y
                norm = np.sqrt(delta_x**2 + delta_y**2)

                if norm != 0:
                    normal_x = delta_y / norm
                    normal_y = -delta_x / norm
                    tangent_x = delta_x / norm
                    tangent_y = delta_y / norm
                else:
                    normal_x = 0
                    normal_y = 0
                    tangent_x = 0
                    tangent_y = 0

                scale_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
                num_large_ticks = 11
                num_small_ticks = 101

                self.draw_scale_on_image(start_x, start_y, end_x, end_y, scale_length, normal_x, normal_y, tangent_x,
                                            tangent_y, line_thickness, num_large_ticks, num_small_ticks)

                cv2.imshow("Video mit Skala", self.first_frame)

                if cv2.waitKey(30) & 0xFF == 27:
                    break
                
    def ResizeWithAspectRatio(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation=inter)
    
    def Crop(self):
        if self.first_frame is not None:
            self.reset_all_buttons()
            if platform.system() == "Darwin":   ### if its a Mac
                self.crop_button.config(relief='sunken',highlightbackground='orange')
            else:
                self.crop_button.config(relief='sunken',bg='orange') 
            #cv2.setWindowProperty("Video", cv2.WND_PROP_TOPMOST, 1)
            cv2.setMouseCallback("Video", self.select_point_crop)

    def select_point_crop(self, event, x, y, flags, param):
        global start_x,start_y,end_x,end_y,scale_length,scale_image,x_shift,y_shift,pos_x,pos_y,lnorm, lstart_x, lstart_y
        if event == cv2.EVENT_MOUSEMOVE:
            if not self.selecting_start_point_crop:
                #if not self.start_end_selected:
                #self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                #ret, self.first_frame = self.cap.read()
                self.first_frame = cached_frames[current_frame].copy()
                
                if 1==1:
                    #height_px = int(0.9*root.winfo_screenheight())
                    #if self.first_frame.shape[0]>height_px and platform.system() != "Darwin":
                    #    tempimage = self.ResizeWithAspectRatio(self.first_frame, height=height_px)
                    #    self.first_frame=tempimage
                    cv2.rectangle(self.first_frame,(self.crop_x0,self.crop_y0),(x,y),(0,120,255),3)
                    if scale_added:
                        self.overlay_image(self.first_frame)
                    if timestamp_added:
                        frame1=self.draw_timestamp(self.first_frame)
                        self.first_frame=frame1.copy()
                        #self.draw_timestamp(self.first_frame)
                    cv2.imshow("Video", self.first_frame)
                    cv2.waitKey(1)
                        
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.selecting_start_point_crop:
                self.crop_x0=x
                self.crop_y0=y
                
                #Ausgewählten Punkt anzeigen
                #cv2.circle(self.first_frame,(x,y), 5, (0,0,255), -1)
                if scale_added:
                    self.overlay_image(self.first_frame)
                if timestamp_added:
                    #self.draw_timestamp(self.first_frame)
                    frame1=self.draw_timestamp(self.first_frame)
                    self.first_frame=frame1.copy()
                cv2.imshow("Video", self.first_frame)
                self.selecting_start_point_crop = False
            else:
                self.crop_x1=x
                self.crop_y1=y
                if self.crop_x1<self.crop_x0:
                    self.crop_x0,self.crop_x1=self.crop_x1,self.crop_x0
                if self.crop_y1<self.crop_y0:
                    self.crop_y0,self.crop_y1=self.crop_y1,self.crop_y0
                    
                    
                self.selecting_start_point_crop = True
                #Ausgewählten Punkt anzeigen
                #cv2.circle(self.first_frame,(x,y), 5, (0,0,255), -1)
                cv2.rectangle(self.first_frame,(self.crop_x0,self.crop_y0),(x,y),(0,120,255),3)
                if scale_added:
                    self.overlay_image(self.first_frame)
                if timestamp_added:
                    frame1=self.draw_timestamp(self.first_frame)
                    self.first_frame=frame1.copy()
                    #self.draw_timestamp(self.first_frame)
                cv2.imshow("Video", self.first_frame)
                self.start_end_selected_crop=True


    def set_frame_min(self):
        self.frame_min=self.video_slider.get()
        self.frame_min_label.config(text=f"Erstes Bild bei Nr.: {self.frame_min}")
        self.frame_max_label.config(text=f"Letztes Bild bei Nr.: {self.frame_max}")
        

    def set_frame_max(self):
        self.frame_max=self.video_slider.get()
        self.frame_min_label.config(text=f"Erstes Bild bei Nr.: {self.frame_min}")
        self.frame_max_label.config(text=f"Letztes Bild bei Nr.: {self.frame_max}")
        




    def quit(self):
        #global addedscale
        if self.cap is not None:
            self.cap.release()
            
        print("On exit",scale_added)
        #addedscale=self.scale_added
        addedscale=True
        #current_frame=self.current_frame
        self.root.destroy()
        cv2.destroyAllWindows()
        
        self.main_menu_callback()
        #if self.cap is not None:
         #   self.cap.release()
            #cv2.destroyAllWindows()
          #  self.main_menu_callback()
        #if self.first_frame is not None:
         #   cv2.destroyWindow("Video")
        #cv2.destroyAllWindows()
        #self.root.destroy()
        #self.main_menu_callback()
        #self.root.quit()




############################################################################################################################################
############################################################################################################################################
############## MAINLOOP
############################################################################################################################################
############################################################################################################################################

if __name__ == "__main__":
    root = tk.Tk()
    cached_frames=None
    total_frames=0
    current_frame=0
    fps=30.
    fps_native=30.
    t0_frame=0
    t0_time=0
    start_point=None
    scale_added=False
    imageflag = False

    app = ScriptSelector(root)
    root.mainloop()

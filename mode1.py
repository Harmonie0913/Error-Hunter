import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
from PIL import Image, ImageTk
from test import on_choose_traindata_button_click
from test import on_choose_testdata_button_click
from test import train_and_visualize
from test import save_model
from test import entwickeln_and_visualize
from test import load_ai_model
from test import StdoutRedirector
from server import connect
from server import disconnect
from server import server_stop
from server import create_TCP_csv
from server import sotieren
from  server import tcp_train_and_visualize
from server import tcp_save_model
from server import save_csv
from server import open_existing_csv

trained_model = None

class Mode1UIManager:
    
    def __init__(self,window):
        #self.parent = parent
        #self.window = tk.Toplevel(parent)
        self.window = tk.Toplevel(window)
        self.window.iconbitmap(r"C:\Users\yangx\OneDrive - KUKA AG\EH\KI.ico")
        self.current_mode = 'Training'
        self.server_socket = None
        self.selector = None
       
    def insert_image(self):
        image_path = r"C:\Users\yangx\OneDrive - KUKA AG\EH\KUKA_Logo.png"  # 替换为你的图片路径
        self.new_size = (140, 24)
        # 打开图像并将其转换为 PhotoImage 对象
        original_image = Image.open(image_path)
        resized_image = original_image.resize(self.new_size)  # 调整图像大小
        photo = ImageTk.PhotoImage(resized_image)
        self.image_label = tk.Label(self.frame_top)
        # 在标签中插入图片
        self.image_label.config(image=photo)
        self.image_label.image = photo  # 保持对 PhotoImage 的引用
        self.image_label.pack(side="left", anchor="ne")
    
    def create_Manuell_widgets(self):
        self.current_mode = 'Manuell'
        self.window.title("Train Mode")
        self.window.geometry("735x900")
        menubar = tk.Menu(self.window)
        mode_menu = tk.Menu(menubar, tearoff=0)
        mode_menu.add_command(label="Training", command=self.create_Training_window)
        mode_menu.add_command(label="Manuell", command=self.create_Manuell_window)
        mode_menu.add_command(label="Entwicklung", command=self.create_Entwicklung_window)
        menubar.add_cascade(label="Select Funktion", menu=mode_menu)
        self.window.config(menu=menubar)
        
        #Logo und Name
        self.frame_top = tk.Frame(self.window)
        self.frame_top.pack(side="top", fill="x", pady=0)
        self.insert_image()

        font = ('times', 15, 'bold')
        self.ProgName = tk.Label(self.frame_top, text='Train Mode', font=font, fg='red')
        self.ProgName.pack(side="right")

        #Kurven
        self.fig, (self.ax_loss, self.ax_acc) = plt.subplots(2, 1, figsize=(5, 8), gridspec_kw={'height_ratios': [1, 1]})
        self.ax_loss.set_title('Training Loss')
        self.ax_loss.set_xlabel('Training Steps')
        self.ax_loss.set_ylabel('Loss')

        self.ax_acc.set_title('Validation Accuracy')
        self.ax_acc.set_xlabel('Training Steps')
        self.ax_acc.set_ylabel('Accuracy (%)')

        plt.subplots_adjust(hspace=0.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side="right", fill='both', anchor=None, pady=0)

        #Trainingsdaten Hochladen
        self.frame_top1 = tk.Frame(self.window)
        self.frame_top1.pack(side="top", fill="x", pady=10)
        self.csv_button = tk.Button(self.frame_top1, text="Trainingsdaten hochladen", command=lambda: on_choose_traindata_button_click(self.csv_button,self.Trainsinfo_text))
        self.Trainsinfo = tk.Label(self.frame_top1,text='Trainingsdaten Info')
        self.Trainsinfo_text = tk.Text(self.frame_top1,height=4, width=50, wrap='word')
        self.csv_button.pack()
        self.Trainsinfo.pack()
        self.Trainsinfo_text.pack()
        
        #validerungsdaten Hochladen
        self.Testdaten_button = tk.Button(self.frame_top1, text="Validierungsdaten hochladen", command=lambda: on_choose_testdata_button_click(self.Testinfo_text,self.Testdaten_button))
        self.Testinfo = tk.Label(self.frame_top1,text='Validierungsdaten Info')
        self.Testinfo_text = tk.Text(self.frame_top1,height=4, width=50, wrap='word')
        self.Info = tk.Label(self.frame_top1,text='Die Anzahl der Merkmalen der Validierungsdaten und\nder Trainingsdaten müssen geleich!')
        self.Testdaten_button.pack()
        self.Testinfo.pack()
        self.Testinfo_text.pack()
        self.Info.pack()

        #Einstellung der Hyperparameter
        self.frame_top2 = tk.Frame(self.window)
        self.frame_top2.pack(side="top", fill="x", pady=4)

        self.sample_number_label_1 = tk.Label(self.frame_top2, text="Geben Sie bitte die\nTrainingsparameter ein! ")
        self.sample_number_label_1.pack(side="top")

        self.frame1 = tk.Frame(self.window)
        self.frame1.pack(side="top", fill="x", pady=4)

        self.grenze_label = tk.Label(self.frame1, text="Grenze:")
        self.grenze_label.pack(side="left")

        self.grenze_entry = tk.Entry(self.frame1, width=5)
        self.grenze_entry.pack(side="left")

        self.gerenze_label1 = tk.Label(self.frame1, text="(Standardwert:0.5)")
        self.gerenze_label1.pack(side="left")

        self.frame_rate = tk.Frame(self.window)
        self.frame_rate.pack(side="top", fill="x", pady=4)

        self.rate_label = tk.Label(self.frame_rate, text="Lernrate:")
        self.rate_label.pack(side="left")

        self.rate_entry = tk.Entry(self.frame_rate, width=5)
        self.rate_entry.pack(side="left")

        self.rate_label1 = tk.Label(self.frame_rate, text="(Standardwert:0.01)")
        self.rate_label1.pack(side="left")

        self.frame2 = tk.Frame(self.window)
        self.frame2.pack(side="top", fill="x", pady=4)

        self.epoch_label = tk.Label(self.frame2, text="Anzahl des Trainingsrunden:")
        self.epoch_label.pack(side="left")

        self.epoch_entry = tk.Entry(self.frame2, width=5)
        self.epoch_entry.pack(side="left")
        
        self.st_frame = tk.Frame(self.window)
        self.st_frame.pack(side="top", fill="x", pady=0)
        
        self.epoch_label1 = tk.Label(self.st_frame, text="(Standardwert:2~5)")
        self.epoch_label1.pack(side="top")

        self.frame3 = tk.Frame(self.window)
        self.frame3.pack(side="top", fill="x", pady=10)

        self.train_button = ttk.Button(self.frame3, text="Start Training", command=lambda:train_and_visualize(self.epoch_entry,self.rate_entry,self.grenze_entry,self.ax_loss,self.ax_acc,self.canvas,self.text,self.ge_Text))
        self.train_button.pack(side="top")

        self.ge_frame = tk.Frame(self.window)
        self.ge_frame.pack(side="top", fill="x", pady=4)

        self.rate_label = tk.Label(self.ge_frame, text="Die Genauigkeit des Modells:")
        self.rate_label.pack(side="left")

        self.ge_Text = tk.Text(self.ge_frame,height=1, width=6)
        self.ge_Text.pack(side="left")

        self.frame4 = tk.Frame(self.window)
        self.frame4.pack(side="top", fill="x", pady=10)

        self.save_button = tk.Button(self.frame4, text="Save Model", command=save_model)
        self.save_button.pack()

        self.frame5 = tk.Frame(self.window)
        self.frame5.pack(side="top", fill="x", pady=10)

        self.text = tk.Text(self.frame5, height=15, width=40, state='normal', wrap='word')
        self.text.pack(side='top')

        self.Exit_button = tk.Button(self.frame5, text="Quit", command=self.window.quit)
        self.Exit_button.pack(side="bottom")
        
        # 创建输出重定向器并重定向标准输出流
        stdout_redirector = StdoutRedirector(self.text)
        sys.stdout = stdout_redirector

    def create_Training_widgets(self):
        self.current_mode = 'Training'
        self.window.title("Training")
        self.window.geometry("735x950")
        menubar = tk.Menu(self.window)
        mode_menu = tk.Menu(menubar, tearoff=0)
        mode_menu.add_command(label="Training", command=self.create_Training_window)
        mode_menu.add_command(label="Manuell", command=self.create_Manuell_window)
        mode_menu.add_command(label="Entwicklung", command=self.create_Entwicklung_window)
        menubar.add_cascade(label="Select Funktion", menu=mode_menu)
        self.window.config(menu=menubar)
        
        #Logo und Name
        self.frame_top = tk.Frame(self.window)
        self.frame_top.pack(side="top", fill="x", pady=0)
        self.insert_image()

        font = ('times', 15, 'bold')
        self.ProgName = tk.Label(self.frame_top, text='Training', font=font, fg='red')
        self.ProgName.pack(side="right")

        #Kurven
        self.fig, (self.ax_loss, self.ax_acc) = plt.subplots(2, 1, figsize=(5, 8), gridspec_kw={'height_ratios': [1, 1]})
        self.ax_loss.set_title('Training Loss')
        self.ax_loss.set_xlabel('Training Steps')
        self.ax_loss.set_ylabel('Loss')

        self.ax_acc.set_title('Validation Accuracy')
        self.ax_acc.set_xlabel('Training Steps')
        self.ax_acc.set_ylabel('Accuracy (%)')

        plt.subplots_adjust(hspace=0.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side="right", fill='both', anchor=None, pady=0)

        #TCP
        self.frame_Host = tk.Frame(self.window)
        self.frame_Host.pack(side="top", fill="x", pady=10)
        self.frame_Port = tk.Frame(self.window)
        self.frame_Port.pack(side="top", fill="x", pady=10)
    
        self.Host_Label = tk.Label(self.frame_Host,text='SERVER HOST: ')
        self.Host_Entry = tk.Entry(self.frame_Host)
        self.Port_Label = tk.Label(self.frame_Port,text='SERVER PORT: ')
        self.Port_Entry = tk.Entry(self.frame_Port)
        
        self.Host_Label.pack(side="left")
        self.Host_Entry.pack(side="top")
        self.Port_Label.pack(side="left")
        self.Port_Entry.pack(side="top")

        self.frame_Connectbutton=tk.Frame(self.window)
        self.frame_Connectbutton.pack(side="top", fill="x", pady=5)
        self.frame_Connectbutton2=tk.Frame(self.window)
        self.frame_Connectbutton2.pack(side="top", fill="x", pady=5)
        self.connect_button = tk.Button(self.frame_Connectbutton, text="Start", command=lambda:connect(self.connect_button, self.disconnect_button, self.Host_Entry, self.Port_Entry,self.Info_Datensatz_text,self.Kurvestatus_text,self.Kurvestatus_text2))
        self.disconnect_button = tk.Button(self.frame_Connectbutton, text="Stop", command=lambda:disconnect(self.connect_button, self.disconnect_button,self.Info_Datensatz_text))
        
        self.connect_button.pack(side="left")
        self.disconnect_button.pack(side="top")        
        
        self.tcpcsv_button = tk.Button(self.frame_Connectbutton2, text="New CSV", command=lambda:create_TCP_csv(self.Info_Datensatz_text))
        self.tcpcsv_button.pack(side="left")
        self.tcpcsv_button = tk.Button(self.frame_Connectbutton2, text="Open CSV", command=lambda:open_existing_csv(self.Info_Datensatz_text,self.Kurvestatus_text,self.Kurvestatus_text2))
        self.tcpcsv_button.pack(side="top")

        self.frame_kurve = tk.Frame(self.window)
        self.frame_kurve.pack(side="top", fill="x", pady=5)
        self.frame_kurve_type = tk.Frame(self.window)
        self.frame_kurve_type.pack(side="top", fill="x", pady=5)

        self.Info_Kurvestatus_Label = tk.Label(self.frame_kurve,text='Info über aktuelle Kurve')
        self.Kurvestatus_Label = tk.Label(self.frame_kurve_type,text='Type:')
        self.Kurvestatus_text = tk.Text(self.frame_kurve_type,height=1, width=6, wrap='word')
        self.Kurvestatus_Label2 = tk.Label(self.frame_kurve_type,text='Test Nummer:')
        self.Kurvestatus_text2 = tk.Text(self.frame_kurve_type,height=1, width=6, wrap='word')
       
        self.Info_Kurvestatus_Label.pack()
        self.Kurvestatus_Label.pack(side="left")
        self.Kurvestatus_text.pack(side="left")
        self.Kurvestatus_text2.pack(side="right")
        self.Kurvestatus_Label2.pack(side="right")
        
      
        #Info über Datensatz und sortieren
        self.frame_Info=tk.Frame(self.window)
        self.frame_Info.pack(side="top", fill="x", pady=5)
        self.Info_Datensatz_Label = tk.Label(self.frame_Info,text='Info über Datensatz')
        self.Info_Datensatz_text = tk.Text(self.frame_Info,height=7, width=50, wrap='word')
        self.sortieren_button = tk.Button(self.frame_Info, text="sortieren", command=lambda:sotieren(self.Info_Datensatz_text))
        self.save_csv_button = tk.Button(self.frame_Info, text="save", command=lambda:save_csv(self.Info_Datensatz_text))
        self.Info_Datensatz_Label.pack()
        self.Info_Datensatz_text.pack()
        self.sortieren_button.pack(side="left")
        self.save_csv_button.pack(side="right")

        #Einstellung der Hyperparameter
        self.frame_top2 = tk.Frame(self.window)
        self.frame_top2.pack(side="top", fill="x", pady=4)

        self.sample_number_label_1 = tk.Label(self.frame_top2, text="Geben Sie bitte die\nTrainingsparameter ein! ")
        self.sample_number_label_1.pack(side="top")

        self.frame1 = tk.Frame(self.window)
        self.frame1.pack(side="top", fill="x", pady=4)

        self.rate_label = tk.Label(self.frame1, text="Lernrate:")
        self.rate_label.pack(side="left")

        self.rate_entry = tk.Entry(self.frame1, width=5)
        self.rate_entry.pack(side="left")

        self.rate_label1 = tk.Label(self.frame1, text="(Standardwert:0.01)")
        self.rate_label1.pack(side="left")

        self.frame2 = tk.Frame(self.window)
        self.frame2.pack(side="top", fill="x", pady=4)

        self.epoch_label = tk.Label(self.frame2, text="Anzahl des Trainingsrunden:")
        self.epoch_label.pack(side="left")

        self.epoch_entry = tk.Entry(self.frame2, width=5)
        self.epoch_entry.pack(side="left")
        
        self.st_frame = tk.Frame(self.window)
        self.st_frame.pack(side="top", fill="x", pady=0)
        
        self.epoch_label1 = tk.Label(self.st_frame, text="(Standardwert:2~5)")
        self.epoch_label1.pack(side="top")

        self.frame3 = tk.Frame(self.window)
        self.frame3.pack(side="top", fill="x", pady=5)

        self.train_button = ttk.Button(self.frame3, text="Start Training", command=lambda:tcp_train_and_visualize(self.epoch_entry,self.rate_entry,self.ax_loss,self.ax_acc,self.canvas,self.text,self.ge_Text))
        self.train_button.pack(side="top")

        self.ge_frame = tk.Frame(self.window)
        self.ge_frame.pack(side="top", fill="x", pady=4)

        self.rate_label = tk.Label(self.ge_frame, text="Die Genauigkeit des Modells:")
        self.rate_label.pack(side="left")

        self.ge_Text = tk.Text(self.ge_frame,height=1, width=6)
        self.ge_Text.pack(side="left")

        self.frame4 = tk.Frame(self.window)
        self.frame4.pack(side="top", fill="x", pady=5)

        self.save_button = tk.Button(self.frame4, text="Save Model", command=tcp_save_model)
        self.save_button.pack()

        self.frame5 = tk.Frame(self.window)
        self.frame5.pack(side="top", fill="x", pady=5)

        self.text = tk.Text(self.frame5, height=15, width=40, state='normal', wrap='word')
        self.text.pack(side='top')

        self.Exit_button = tk.Button(self.frame5, text="Quit", command=self.window.quit)
        self.Exit_button.pack(side="bottom")
        
        # 创建输出重定向器并重定向标准输出流
        stdout_redirector = StdoutRedirector(self.text)
        sys.stdout = stdout_redirector

    def create_Entwicklung_widgets(self):
        self.current_mode = 'Entwicklung'
        self.window.title("Entwicklung")
        self.window.geometry("735x980")
        menubar = tk.Menu(self.window)
        mode_menu = tk.Menu(menubar, tearoff=0)
        mode_menu.add_command(label="Training", command=self.create_Training_window)
        mode_menu.add_command(label="Manuell", command=self.create_Manuell_window)
        mode_menu.add_command(label="Entwicklung", command=self.create_Entwicklung_window)
        menubar.add_cascade(label="Select Funktion", menu=mode_menu)
        self.window.config(menu=menubar)
        
        #Logo und Name
        self.frame_top = tk.Frame(self.window)
        self.frame_top.pack(side="top", fill="x", pady=0)
        self.insert_image()

        font = ('times', 15, 'bold')
        self.ProgName = tk.Label(self.frame_top, text='Training', font=font, fg='red')
        self.ProgName.pack(side="right")

        #Training Kurven
        self.fig, (self.ax_loss, self.ax_acc) = plt.subplots(2, 1, figsize=(5, 8), gridspec_kw={'height_ratios': [1, 1]})
        self.ax_loss.set_title('Training Loss')
        self.ax_loss.set_xlabel('Training Steps')
        self.ax_loss.set_ylabel('Loss')

        self.ax_acc.set_title('Validation Accuracy')
        self.ax_acc.set_xlabel('Training Steps')
        self.ax_acc.set_ylabel('Accuracy (%)')

        plt.subplots_adjust(hspace=0.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side="right", fill='both', anchor=None, pady=0)

        #Entwicklung   Load Model
        self.frame_Kern = tk.Frame(self.window)
        self.frame_Kern.pack(side="top", fill="x", pady=10)        
        self.load_model_button = tk.Button(self.frame_Kern, text="Load AI Model", command=lambda: load_ai_model(self.load_model_button,self.KI_text))
        self.load_model_button.pack()
        self.KI_text = tk.Text(self.window, height=2, width=50, wrap="word")
        self.KI_text.pack()       


        #TCP
        self.frame_Host = tk.Frame(self.window)
        self.frame_Host.pack(side="top", fill="x", pady=10)
        self.frame_Port = tk.Frame(self.window)
        self.frame_Port.pack(side="top", fill="x", pady=10)
    
        self.Host_Label = tk.Label(self.frame_Host,text='SERVER HOST: ')
        self.Host_Entry = tk.Entry(self.frame_Host)
        self.Port_Label = tk.Label(self.frame_Port,text='SERVER PORT: ')
        self.Port_Entry = tk.Entry(self.frame_Port)
        
        self.Host_Label.pack(side="left")
        self.Host_Entry.pack(side="top")
        self.Port_Label.pack(side="left")
        self.Port_Entry.pack(side="top")

        self.frame_Connectbutton=tk.Frame(self.window)
        self.frame_Connectbutton.pack(side="top", fill="x", pady=10)
        self.connect_button = tk.Button(self.frame_Connectbutton, text="Start", command=lambda:connect(self.connect_button, self.disconnect_button, self.Host_Entry, self.Port_Entry,self.Info_Datensatz_text))
        self.disconnect_button = tk.Button(self.frame_Connectbutton, text="Stop", command=lambda:disconnect(self.connect_button, self.disconnect_button,self.Info_Datensatz_text))
        self.connect_button.pack(side="left")
        self.disconnect_button.pack(side="top")        
        
        #Info über Datensatz und sortieren
        self.frame_Info=tk.Frame(self.window)
        self.frame_Info.pack(side="top", fill="x", pady=10)
        self.Info_Datensatz_Label = tk.Label(self.frame_Info,text='Info über Datensatz')
        self.Info_Datensatz_text = tk.Text(self.frame_Info,height=7, width=50, wrap='word')
        self.sortieren_button = tk.Button(self.frame_Info, text="sortieren", command=lambda:train_and_visualize(self.epoch_entry,self.rate_entry,self.ax_loss,self.ax_acc,self.canvas,self.text,self.ge_Text))
        self.Info_Datensatz_Label.pack()
        self.Info_Datensatz_text.pack()
        self.sortieren_button.pack()

        #Einstellung der Hyperparameter
        self.frame_top2 = tk.Frame(self.window)
        self.frame_top2.pack(side="top", fill="x", pady=4)

        self.sample_number_label_1 = tk.Label(self.frame_top2, text="Geben Sie bitte die\nTrainingsparameter ein! ")
        self.sample_number_label_1.pack(side="top")

        self.frame1 = tk.Frame(self.window)
        self.frame1.pack(side="top", fill="x", pady=4)

        self.rate_label = tk.Label(self.frame1, text="Lernrate:")
        self.rate_label.pack(side="left")

        self.rate_entry = tk.Entry(self.frame1, width=5)
        self.rate_entry.pack(side="left")

        self.rate_label1 = tk.Label(self.frame1, text="(Standardwert:0.01)")
        self.rate_label1.pack(side="left")

        self.frame2 = tk.Frame(self.window)
        self.frame2.pack(side="top", fill="x", pady=4)

        self.epoch_label = tk.Label(self.frame2, text="Anzahl des Trainingsrunden:")
        self.epoch_label.pack(side="left")

        self.epoch_entry = tk.Entry(self.frame2, width=5)
        self.epoch_entry.pack(side="left")
        
        self.st_frame = tk.Frame(self.window)
        self.st_frame.pack(side="top", fill="x", pady=0)
        
        self.epoch_label1 = tk.Label(self.st_frame, text="(Standardwert:2~5)")
        self.epoch_label1.pack(side="top")

        self.frame3 = tk.Frame(self.window)
        self.frame3.pack(side="top", fill="x", pady=10)

        #Entwicklung     Train
        self.train_button = ttk.Button(self.frame3, text="Start Training", command=lambda:entwickeln_and_visualize(self.epoch_entry,self.rate_entry,self.ax_loss,self.ax_acc,self.canvas,self.text,self.ge_Text))
        self.train_button.pack(side="top")

        self.ge_frame = tk.Frame(self.window)
        self.ge_frame.pack(side="top", fill="x", pady=4)

        self.rate_label = tk.Label(self.ge_frame, text="Die Genauigkeit des Modells:")
        self.rate_label.pack(side="left")

        self.ge_Text = tk.Text(self.ge_frame,height=1, width=6)
        self.ge_Text.pack(side="left")

        self.frame4 = tk.Frame(self.window)
        self.frame4.pack(side="top", fill="x", pady=10)

        self.save_button = tk.Button(self.frame4, text="Save Model", command=save_model)
        self.save_button.pack()

        self.frame5 = tk.Frame(self.window)
        self.frame5.pack(side="top", fill="x", pady=10)

        self.text = tk.Text(self.frame5, height=15, width=40, state='normal', wrap='word')
        self.text.pack(side='top')

        self.Exit_button = tk.Button(self.frame5, text="Quit", command=self.window.quit)
        self.Exit_button.pack(side="bottom")
        
        # 创建输出重定向器并重定向标准输出流
        stdout_redirector = StdoutRedirector(self.text)
        sys.stdout = stdout_redirector


    def destroy_Entwicklung_widgets(self):
        self.frame_top.pack_forget()
        self.image_label.pack_forget()
        self.ProgName.pack_forget()
        self.canvas_widget.pack_forget()
        self.frame_Kern.pack_forget()
        self.load_model_button.pack_forget()
        self.KI_text.pack_forget()

        self.frame_Host.pack_forget()
        self.frame_Port.pack_forget()
        self.Host_Label.pack_forget()
        self.Host_Entry.pack_forget()
        self.Port_Label.pack_forget()
        self.Port_Entry.pack_forget()
        self.frame_Connectbutton.pack_forget()
        self.connect_button.pack_forget()
        self.disconnect_button.pack_forget()
        self.frame_Info.pack_forget()
        self.Info_Datensatz_Label.pack_forget()
        self.Info_Datensatz_text.pack_forget()
        self.sortieren_button.pack_forget()
        self.frame_top2.pack_forget()
        self.sample_number_label_1.pack_forget()
        self.frame1.pack_forget()
        self.rate_label1.pack_forget()
        self.rate_label.pack_forget()
        self.rate_entry.pack_forget()
        self.image_label.pack_forget()
        self.frame2.pack_forget()
        self.epoch_label.pack_forget()
        self.epoch_entry.pack_forget()
        self.st_frame.pack_forget()
        self.epoch_label1.pack_forget()
        self.frame3.pack_forget()
        self.train_button.pack_forget()
        self.ge_frame.pack_forget()
        self.rate_label.pack_forget()
        self.ge_Text.pack_forget()
        self.frame4.pack_forget()
        self.save_button.pack_forget()
        self.frame5.pack_forget()
        self.text.pack_forget()
        self.Exit_button.pack_forget()



    def destroy_Training_widgets(self):
        self.frame_top.pack_forget()
        self.image_label.pack_forget()
        self.ProgName.pack_forget()
        self.canvas_widget.pack_forget()
        self.frame_Host.pack_forget()
        self.frame_Port.pack_forget()
        self.Host_Label.pack_forget()
        self.Host_Entry.pack_forget()
        self.Port_Label.pack_forget()
        self.Port_Entry.pack_forget()
        self.frame_Connectbutton.pack_forget()
        self.connect_button.pack_forget()
        self.disconnect_button.pack_forget()
        self.frame_Info.pack_forget()
        self.Info_Datensatz_Label.pack_forget()
        self.Info_Datensatz_text.pack_forget()
        self.sortieren_button.pack_forget()
        self.frame_top2.pack_forget()
        self.sample_number_label_1.pack_forget()
        self.frame1.pack_forget()
        self.rate_label1.pack_forget()
        self.rate_label.pack_forget()
        self.rate_entry.pack_forget()
        self.image_label.pack_forget()
        self.frame2.pack_forget()
        self.epoch_label.pack_forget()
        self.epoch_entry.pack_forget()
        self.st_frame.pack_forget()
        self.epoch_label1.pack_forget()
        self.frame3.pack_forget()
        self.train_button.pack_forget()
        self.ge_frame.pack_forget()
        self.rate_label.pack_forget()
        self.ge_Text.pack_forget()
        self.frame4.pack_forget()
        self.save_button.pack_forget()
        self.frame5.pack_forget()
        self.text.pack_forget()
        self.Exit_button.pack_forget()
        self.frame_kurve_type.pack_forget()
        self.frame_kurve.pack_forget()
        self.Info_Kurvestatus_Label.pack_forget()
        self.Kurvestatus_Label.pack_forget()
        self.Kurvestatus_Label2.pack_forget()
        self.Kurvestatus_text.pack_forget()
        self.Kurvestatus_text2.pack_forget()
        self.frame_Connectbutton2.pack_forget()



    def destroy_Manuell_widgets(self):
        self.csv_button.pack_forget()
        self.sample_number_label_1.pack_forget()
        self.Trainsinfo.pack_forget()
        self.Trainsinfo_text.pack_forget()
        self.Testdaten_button.pack_forget()
        self.Testinfo.pack_forget()
        self.Testinfo_text.pack_forget()
        self.Info.pack_forget()
        self.rate_label.pack_forget()
        self.rate_entry.pack_forget()
        self.rate_label1.pack_forget()
        self.epoch_label.pack_forget()
        self.epoch_entry.pack_forget()
        self.train_button.pack_forget()
        self.save_button.pack_forget()
        self.text.pack_forget()
        self.canvas_widget.pack_forget()
        self.st_frame.pack_forget()
        self.frame1.pack_forget()
        self.frame2.pack_forget()
        self.frame3.pack_forget()
        self.frame4.pack_forget()
        self.frame5.pack_forget()
        self.frame_top1.pack_forget()
        self.frame_top2.pack_forget()
        self.frame_top.pack_forget()
        self.epoch_label1.pack_forget()
        self.Exit_button.pack_forget()
        self.image_label.pack_forget()
        self.ProgName.pack_forget()
        self.ge_frame.pack_forget()
        self.ge_Text.pack_forget()
        self.canvas_widget.pack_forget()
        self.frame_rate.pack_forget()
        self.grenze_entry.pack_forget()
        self.grenze_label.pack_forget()
        self.gerenze_label1.pack_forget()
    def create_Training_window(self):
        if  self.current_mode == 'Manuell':
            self.destroy_Manuell_widgets()
            self.create_Training_widgets()
        
        elif  self.current_mode == 'Entwicklung':
              server_stop(self.server_socket, self.selector)
              self.destroy_Entwicklung_widgets()
              self.create_Training_widgets()
        else: 
            self.current_mode = 'Training'

    def create_Manuell_window(self):
        if  self.current_mode == 'Training':
            server_stop(self.server_socket, self.selector)
            self.destroy_Training_widgets()
            self.create_Manuell_widgets()
        
        elif  self.current_mode == 'Entwicklung':
              server_stop(self.server_socket, self.selector)
              self.destroy_Entwicklung_widgets()
              self.create_Manuell_widgets()
        else: 
            self.current_mode = 'Manuell'

    def create_Entwicklung_window(self):
        if  self.current_mode == 'Training':
            server_stop(self.server_socket, self.selector)
            self.destroy_Training_widgets()
            self.create_Entwicklung_widgets()
        
        elif  self.current_mode == 'Manuell':

              self.destroy_Manuell_widgets()
              self.create_Entwicklung_widgets()
        else: 
            self.current_mode = 'Entwicklung'


        
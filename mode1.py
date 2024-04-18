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
from test import StdoutRedirector
import torch
import numpy as np
from torch.utils.data import Dataset

trained_model = None

class WineDataset(Dataset):

    def __init__(self,csv_file_path):
        # Initialize data, download, etc.
        # read with numpy or pandas
        xy = np.loadtxt(csv_file_path, delimiter=',', dtype=np.float32, skiprows=1) 
        self.n_samples = xy.shape[0]
        
        # here the first column is the class label, the rest are the features
        self.x_data = torch.from_numpy(xy[:, 1:751]) # size [n_samples, n_features]
        self.y_data = torch.from_numpy(xy[:, 0]).to(torch.long)


    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]


    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


class Mode1UIManager:
    
    def __init__(self,window):
        #self.parent = parent
        #self.window = tk.Toplevel(parent)
        self.window = tk.Toplevel(window)
        self.window.iconbitmap(r"C:\Users\yangx\OneDrive - KUKA AG\EH\KI.ico")
        #self.create_widgets()
        self.frame_top = tk.Frame(self.window)
        self.frame_top.pack(side="top", fill="x", pady=0)
       
    def insert_image(self,new_size):
        image_path = r"C:\Users\yangx\OneDrive - KUKA AG\EH\KUKA_Logo.png"  # 替换为你的图片路径

        # 打开图像并将其转换为 PhotoImage 对象
        original_image = Image.open(image_path)
        resized_image = original_image.resize(new_size)  # 调整图像大小
        photo = ImageTk.PhotoImage(resized_image)
        self.image_label = tk.Label(self.frame_top)
        # 在标签中插入图片
        self.image_label.config(image=photo)
        self.image_label.image = photo  # 保持对 PhotoImage 的引用
        self.image_label.pack(side="left", anchor="ne")

    def create_widgets(self):
        self.window.title("Train Mode")
        self.window.geometry("715x848")
        
               # 插入图像
        new_size = (140, 24)
        self.insert_image(new_size)
        font = ('times', 15, 'bold')
        self.ProgName = tk.Label(self.frame_top, text='Train Mode', font=font, fg='red')
        self.ProgName.pack(side="right")

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

        self.frame_top1 = tk.Frame(self.window)
        self.frame_top1.pack(side="top", fill="x", pady=10)

        self.csv_button = tk.Button(self.frame_top1, text="Trainingsdaten hochladen", command=lambda: on_choose_traindata_button_click(self.csv_button,self.Trainsinfo_text))
        self.Trainsinfo = tk.Label(self.frame_top1,text='Trainingsdaten Info')
        self.Trainsinfo_text = tk.Text(self.frame_top1,height=4, width=50, wrap='word')
        self.csv_button.pack()
        self.Trainsinfo.pack()
        self.Trainsinfo_text.pack()
        


        self.Testdaten_button = tk.Button(self.frame_top1, text="Validierungsdaten hochladen", command=lambda: on_choose_testdata_button_click(self.Testinfo_text,self.Testdaten_button))
        self.Testinfo = tk.Label(self.frame_top1,text='Validierungsdaten Info')
        self.Testinfo_text = tk.Text(self.frame_top1,height=4, width=50, wrap='word')
        self.Info = tk.Label(self.frame_top1,text='Die Anzahl der Merkmalen der Validierungsdaten und\nder Trainingsdaten müssen geleich!')
        self.Testdaten_button.pack()
        self.Testinfo.pack()
        self.Testinfo_text.pack()
        self.Info.pack()

        self.frame_top2 = tk.Frame(self.window)
        self.frame_top2.pack(side="top", fill="x", pady=4)

        self.sample_number_label_1 = tk.Label(self.frame_top2, text="Geben Sie bitte die\nTrainingsparameter ein! ")
        self.sample_number_label_1.pack(side="top")

        # self.frame = tk.Frame(self.window)
        # self.frame.pack(side="top", fill="x", pady=4)

        # self.input_label = tk.Label(self.frame, text="input size:")
        # self.input_label.pack(side="left")

        # self.input_entry = tk.Entry(self.frame, width=5)
        # self.input_entry.pack(side="left")

        # self.input_label1 = tk.Label(self.frame, text="(Probengröße)")
        # self.input_label1.pack(side="left")

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

        self.train_button = ttk.Button(self.frame3, text="Start Training", command=lambda:train_and_visualize(self.epoch_entry,self.rate_entry,self.ax_loss,self.ax_acc,self.canvas,self.text,self.ge_Text))
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

    # def destroy_widgets(self):
    #     self.csv_button.pack_forget()
    #     self.sample_number_label_1.pack_forget()
    #     self.separator_frame.pack_forget()
    #     self.sample_number_label_2.pack_forget()
    #     self.input_label.pack_forget()
    #     self.input_entry.pack_forget()
    #     self.input_label1.pack_forget()
    #     self.rate_label.pack_forget()
    #     self.rate_entry.pack_forget()
    #     self.epoch_label.pack_forget()
    #     self.epoch_entry.pack_forget()
    #     self.train_button.pack_forget()
    #     self.save_button.pack_forget()
    #     self.text.pack_forget()
    #     self.canvas_widget.pack_forget()
    #     self.frame.pack_forget()
    #     self.frame1.pack_forget()
    #     self.frame2.pack_forget()
    #     self.frame3.pack_forget()
    #     self.frame4.pack_forget()
    #     self.frame5.pack_forget()
    #     self.frame_top1.pack_forget()
    #     self.frame_top2.pack_forget()
    #     self.frame_top.pack_forget()
    #     self.separator_frame.pack_forget()
    #     self.Exit_button.pack_forget()
    #     self.image_label.pack_forget()
    #     self.ProgName.pack_forget()



        
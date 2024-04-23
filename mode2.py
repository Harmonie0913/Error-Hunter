import time
import tkinter as tk
from tkinter import *
import pandas as pd
import numpy as np
from PIL import Image, ImageTk
from tkinter import ttk
import torch.nn.functional as F
from tkinter import filedialog
import time
import pandas as pd
import numpy as np
from torch import nn, load, tensor, max
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from test import load_ai_model



class MyFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyFNN, self).__init__()
        self.l1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.relu = nn.PReLU()
        self.l2 = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

# # 创建 FNN 模型实例
# input_size = test_size  # 输入特征数量
hidden_size = 20  # 隐藏层大小
output_size = 2  # 输出大小
# model = MyFNN(input_size, hidden_size, output_size)

class Mode2UIManager:
    def __init__(self, window):
        self.window = tk.Toplevel(window)
        #self.window = window
        self.labels = ['Gerätename', 'Progress', 'Status', 'Fehlermeldung']
        self.window.iconbitmap(r"C:\Users\yangx\OneDrive - KUKA AG\EH\KI.ico")
        self.window.geometry("600x685")
        self.current_mode = 'Automatisch'
        #self.create_widgets()
    def insert_image(self,new_size):
        image_path = r"C:\Users\yangx\OneDrive - KUKA AG\EH\KUKA_Logo.png"  # 替换为你的图片路径

        # 打开图像并将其转换为 PhotoImage 对象
        original_image = Image.open(image_path)
        resized_image = original_image.resize(new_size)  # 调整图像大小
        photo = ImageTk.PhotoImage(resized_image)
        self.image_label = tk.Label(self.window)
        # 在标签中插入图片
        self.image_label.config(image=photo)
        self.image_label.image = photo  # 保持对 PhotoImage 的引用
        self.image_label.grid(row=0, column=0,sticky="w",pady=9)
   
    def create_Manuell_widgets(self):
        self.window.title("Test Mode")
        #self.window.geometry("700x800")
        self.current_mode = 'Manuell'
        menubar = tk.Menu(self.window)
        mode_menu = tk.Menu(menubar, tearoff=0)
        mode_menu.add_command(label="Automatisch", command=self.create_Automatisch_window)
        mode_menu.add_command(label="Manuell", command=self.create_Manuell_window)
        menubar.add_cascade(label="Select Funktion", menu=mode_menu)
        self.window.config(menu=menubar)
        
        # 插入图像
        new_size = (140, 24)
        self.insert_image(new_size)
        text = "Test Mode"
        font = ('times', 15, 'bold')
        color = "red"
        self.ProgName = tk.Label(self.window, text=text, font=font, fg=color)
        self.ProgName.grid(row=0, column=0, columnspan=5,pady=9,sticky="E")

        
        # 表头
        self.label_widgets = []
        
        for col, label_text in enumerate(self.labels, start=0):
            label = tk.Label(self.window, text=label_text, relief=tk.RIDGE, width=15)
            label.grid(row=4, column=col)
            self.label_widgets.append(label)
                
        # 数据初始化
        self.motor = 'Ventil1'
        self.progress_values = {self.motor: 0}
        self.status_values = {self.motor: 'No Test'}
        self.error_values = {self.motor: ''}

        # 创建进度条、状态和错误消息的字典
        self.status_labels = {}
        self.error_labels = {}

        # 填充表格数据
        self.row = 5
        self.motor_label = tk.Label(self.window, text=self.motor)
        self.motor_label.grid(row=self.row, column=0)

        # 状态
        self.status_labels[self.motor] = tk.Label(self.window, text=self.status_values[self.motor], width=15)
        self.status_labels[self.motor].grid(row=self.row, column=2)

        # 错误消息
        self.error_labels[self.motor] = tk.Label(self.window, text=self.error_values[self.motor], width=15)
        self.error_labels[self.motor].grid(row=self.row, column=3)

        #csv auswählen
        self.generate_button1 = Button(self.window, text="Select Daten", command=lambda: self.on_choose_csv_button_click(self.generate_button1))
        self.generate_button1.grid(row=1,column=3,pady=9,sticky=E)


        self.text_Field = Text(self.window, height=2, width=50, wrap=WORD)
        self.text_Field.grid(row=1, column=0, pady=9, columnspan=3,sticky=W)


        # 新增 on_generate_button_click 函数
        self.Kern_button = Button(self.window, text="Select KI Modell", command=lambda: load_ai_model(self.Kern_button,self.KI_text))
        self.Kern_button.grid(row=2, column=3,pady=9,sticky=E)


        self.KI_text = Text(self.window, height=2, width=50, wrap=WORD)
        self.KI_text.grid(row=2, column=0, pady=9, columnspan=3,sticky=W)


        # 新增 on_generate_button_click 函数
        self.generate_button2 = Button(self.window, text="CHECK IT NOW!", command=self.on_generate_button_click)
        self.generate_button2.grid(row=3, column=3,pady=9,sticky=E)


        # 新增文本框用于输入样本号
        self.sample_number_label = Label(self.window, text="Please enter your sample number:")
        self.sample_number_label.grid(row=3, column=0,pady=5,sticky=W)
        #enter
        self.sample_number_entry = tk.Entry(self.window,width=10)
        self.sample_number_entry.grid(row=3, column=2,pady=5,sticky=W)

        self.Exit_button = Button(self.window, text="Quit", command=self.window.quit)
        self.Exit_button.grid(row=10, column=0, columnspan=6, pady=19)

        self.fig = plt.figure(figsize=(6, 4))
        self.fig.suptitle('Visualisierung der Testdaten')
        self.fig.gca().set_xlabel('Zeit')
        self.fig.gca().set_ylabel('Druck')
        # 将绘图对象添加到Tkinter窗口中
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=8, column=0,columnspan=5,pady=2)


    def create_Automatisch_widgets(self):
        self.window.title("Test Mode")
        #self.window.geometry("700x800")
        self.current_mode = 'Automatisch'
        menubar = tk.Menu(self.window)
        mode_menu = tk.Menu(menubar, tearoff=0)
        mode_menu.add_command(label="Automatisch", command=self.create_Automatisch_window)
        mode_menu.add_command(label="Manuell", command=self.create_Manuell_window)
        menubar.add_cascade(label="Select Funktion", menu=mode_menu)
        self.window.config(menu=menubar)
        
        # 插入图像
        new_size = (140, 24)
        self.insert_image(new_size)
        text = "Test Mode"
        font = ('times', 15, 'bold')
        color = "red"
        self.ProgName = tk.Label(self.window, text=text, font=font, fg=color)
        self.ProgName.grid(row=0, column=0, columnspan=5,pady=9,sticky="E")

        
        # 表头
        self.label_widgets = []
        
        for col, label_text in enumerate(self.labels, start=0):
            label = tk.Label(self.window, text=label_text, relief=tk.RIDGE, width=15)
            label.grid(row=4, column=col)
            self.label_widgets.append(label)
                
        # 数据初始化
        self.motor = 'Ventil1'
        self.progress_values = {self.motor: 0}
        self.status_values = {self.motor: 'No Test'}
        self.error_values = {self.motor: ''}

        # 创建进度条、状态和错误消息的字典
        self.status_labels = {}
        self.error_labels = {}

        # 填充表格数据
        self.row = 5
        self.motor_label = tk.Label(self.window, text=self.motor)
        self.motor_label.grid(row=self.row, column=0)

        # 状态
        self.status_labels[self.motor] = tk.Label(self.window, text=self.status_values[self.motor], width=15)
        self.status_labels[self.motor].grid(row=self.row, column=2)

        # 错误消息
        self.error_labels[self.motor] = tk.Label(self.window, text=self.error_values[self.motor], width=15)
        self.error_labels[self.motor].grid(row=self.row, column=3)

        #TCP
        self.Host_Label = tk.Label(self.window,text='SERVER HOST: ')
        self.Host_Entry =tk.Entry(self.window,width=20)
        self.Port_Label = tk.Label(self.window,text='SERVER PORT: ')
        self.Port_Entry =tk.Entry(self.window,width=20)
       
        self.Host_Label.grid(row=1, column=0, pady=5,padx=2,sticky=W)
        self.Host_Entry.grid(row=1, column=1, pady=5,padx=2,sticky=W)
        self.Port_Label.grid(row=1, column=2, pady=5,padx=2,sticky=W)
        self.Port_Entry.grid(row=1, column=3, pady=5,padx=2,sticky=W)


        self.connect_button = tk.Button(self.window,text='Connect',command=lambda: load_ai_model(self.Kern_button,self.KI_text))
        self.disconnect_button = tk.Button(self.window,text='Disconnect',command=lambda: load_ai_model(self.Kern_button,self.KI_text),state=tk.DISABLED)
        self.connect_button.grid(row=2, column=1, pady=5,columnspan=2,sticky=W)
        self.disconnect_button.grid(row=2, column=2, pady=5,columnspan=2,sticky=W)      
       
        #Model auswählen
        self.Kern_button = Button(self.window, text="Select KI Modell", command=lambda: load_ai_model(self.Kern_button,self.KI_text))
        self.Kern_button.grid(row=3, column=3,pady=9,sticky=E)


        self.KI_text = Text(self.window, height=2, width=50, wrap=WORD)
        self.KI_text.grid(row=3, column=0, pady=9, columnspan=3,sticky=W)



        self.Exit_button = Button(self.window, text="Quit", command=self.window.quit)
        self.Exit_button.grid(row=10, column=0, columnspan=6, pady=19)

        self.fig = plt.figure(figsize=(6, 4))
        self.fig.suptitle('Visualisierung der Testdaten')
        self.fig.gca().set_xlabel('Zeit')
        self.fig.gca().set_ylabel('Druck')
        # 将绘图对象添加到Tkinter窗口中
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=8, column=0,columnspan=5,pady=2)


    
    def classify_wine(self):
        global test_size
        try:
            if self.sample_number < 2:
                    raise ValueError("Sample number is out of bounds.")
            
            self.csv_file_path = r"{}".format(self.csv_file_path)
            # CSV 文件路径
            data = pd.read_csv(self.csv_file_path)

            # 选择特定行的数据
            self.input_data_pandas = data.iloc[self.sample_number-2, 1:].values
            self.test_size = len(self.input_data_pandas)
            
            # 转换为 PyTorch tensor
            input_tensor = tensor(self.input_data_pandas.astype(np.float32))

            # 加载训练好的模型
            self.ki_file_path = r"{}".format(self.ki_file_path)
            loaded_parameters = load(self.ki_file_path)
            loaded_model = MyFNN(self.test_size, hidden_size, output_size)
            loaded_model.load_state_dict(loaded_parameters)

            # 记录开始时间
            start_time = time.time()

            # 使用加载的模型进行推断
            output = loaded_model(input_tensor)

            # 记录结束时间
            end_time = time.time()

            # 使用 softmax 获取概率并找到预测的索引
            probabilities = F.softmax(output, dim=0)
            _, predicted_index = max(probabilities, 0)
            predicted_class = predicted_index.item()   # 假设类别索引从 1 开始

            # 类别映射字典
            class_mapping = {
                0: "Das Ventil ist ZU",
                1: "Das Ventil ist AUF!!!",
            }

            # 获取预测类别的文字描述
            predicted_description = class_mapping.get(predicted_class, "Unknown")

            # 计算每一步的执行时间
            loading_data_time = -start_time + time.time()

            inference_time = end_time - start_time

            # 返回 inference_time, predicted_class, predicted_description
            return inference_time, predicted_class, predicted_description,loading_data_time

        except ValueError as ve:
            # 处理异常
            return -1, -1, "Probe existiert nicht", -1


        except Exception as e:

            return -1, -1,"Probe existiert nicht",-1


    # 仿真数据加载
    def simulate_data_loading(self,predicted_class,inference_time):
  
            # 检查是否存在旧的进度条，如果存在则移除它
        if hasattr(self, 'progress_bars') and self.motor in self.progress_bars:
            self.progress_bars[self.motor].grid_forget()
            

        self.progress_bars = {}
        self.s = ttk.Style()  # 在循环之前创建 ttk.Style() 对象

        if predicted_class == -1:
            self.s.theme_use('clam')
            self.s.configure("yellow.Horizontal.TProgressbar", foreground='yellow', background='yellow')
        
        elif predicted_class == 0:
            self.s.theme_use('clam')
            self.s.configure("green.Horizontal.TProgressbar", foreground='green', background='green')

        else:
            self.s.theme_use('clam')
            self.s.configure("red.Horizontal.TProgressbar", foreground='red', background='red')   

        
    # 在循环之外创建进度条
        progress_bar_style = "yellow.Horizontal.TProgressbar" if predicted_class == -1 else \
                        "green.Horizontal.TProgressbar" if predicted_class == 0 else \
                        "red.Horizontal.TProgressbar"

        self.progress_bars[self.motor] = ttk.Progressbar(self.window, style=progress_bar_style, length=100)
        self.progress_bars[self.motor].grid(row=self.row, column=1)

        for i in range(101):
        # 更新进度条的值
            self.progress_bars[self.motor]['value'] = i
            self.window.update_idletasks()
        
        # 更新状态标签
            if predicted_class == 0:
                self.status_labels[self.motor]['text'] = "Processing..."
            else:
                self.status_labels[self.motor]['text'] = "WARNUNG!!!"
        
        # 计算休眠时间
            if predicted_class == -1:
                time.sleep(1e-20)
            else:
                time.sleep(inference_time / 100)


    # 更新错误消息
    def update_fehlermeldung(self,predicted_description):
        self.error_labels[self.motor]['text'] = f"{predicted_description}"

    def Visualisierung(self):
        self.fig.clear()
        self.fig.suptitle('Visualisierung der Testdaten')
        ax = self.fig.gca()
        ax.set_xlabel('Zeit')
        ax.set_ylabel('Druck')
        ax.plot(self.input_data_pandas, color='blue')

                        # 刷新画布
        self.canvas.draw_idle()
        self.canvas.flush_events()

    def on_generate_button_click(self):
        # 获取样本编号
        self.sample_number = int(self.sample_number_entry.get())

        # 调用 classify_wine 函数并获取返回值
        inference_time, predicted_class, predicted_description, loading_data_time = self.classify_wine()
                                                                                                       
        self.Visualisierung()                                                                                    
        
        # 调用 simulate_data_loading 函数
        self.simulate_data_loading(predicted_class,inference_time)

        # 更新错误消息
        self.update_fehlermeldung(predicted_description)


    def on_choose_csv_button_click(self,csv_button):
        
        # 获取用户选择的 CSV 文件路径
        self.csv_file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV Files", "*.csv")])

        # 检查是否选择了文件
        if self.csv_file_path:
            # 将文件路径显示在界面上，或进行其他处理
            csv_button.configure(bg="green")
            #print(f"Selected CSV file: {self.csv_file_path}")
            self.text_Field.delete(1.0, END)  # 清空文本框
            self.text_Field.insert(END, self.csv_file_path)

    def destroy_Manuell_widgets(self):
        self.image_label.grid_forget()
        self.ProgName.grid_forget()
        self.generate_button1.grid_forget()
        self.generate_button2.grid_forget()
        self.Kern_button.grid_forget()
        self.Exit_button.grid_forget()
        self.KI_text.grid_forget()
        self.text_Field.grid_forget()
        self.canvas_widget.grid_forget()
        self.sample_number_entry.grid_forget()
        self.sample_number_label.grid_forget()
        for label in self.label_widgets:
            label.grid_forget()
        self.status_labels[self.motor].grid_forget()
        self.error_labels[self.motor].grid_forget()
        self.motor_label.grid_forget()



    def destroy_Automatisch_widgets(self):
        self.image_label.grid_forget()
        self.ProgName.grid_forget()
        self.Host_Label.grid_forget()
        self.Host_Entry.grid_forget()
        self.Port_Label.grid_forget()
        self.Port_Entry.grid_forget()
        self.connect_button.grid_forget()
        self.disconnect_button.grid_forget()
        self.Kern_button.grid_forget()
        self.Exit_button.grid_forget()
        self.KI_text.grid_forget()
        self.canvas_widget.grid_forget()

        for label in self.label_widgets:
            label.grid_forget()
        self.status_labels[self.motor].grid_forget()
        self.error_labels[self.motor].grid_forget()
        self.motor_label.grid_forget()




    def create_Automatisch_window(self):
        if  self.current_mode == 'Manuell':
            self.destroy_Manuell_widgets()
            self.create_Automatisch_widgets()

        else: 
            self.current_mode = 'Automatisch' 

    def create_Manuell_window(self):
        if  self.current_mode == 'Automatisch':
            self.destroy_Automatisch_widgets()
            self.create_Manuell_widgets()

        else: 
            self.current_mode = 'Manuell'         
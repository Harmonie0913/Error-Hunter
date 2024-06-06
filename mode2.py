import time
import tkinter as tk
from tkinter import *
import pandas as pd
import numpy as np
from PIL import Image, ImageTk
from tkinter import ttk
import torch.nn.functional as F
import torch
from tkinter import filedialog
import time
import pandas as pd
import numpy as np
from torch import nn, load, tensor, max
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from test import load_ai_model
from server import connect
from server import disconnect
from server import server_stop
from server import Test_connect
from server import data_queue
from test import NeuralNet

import server


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
        self.labels = ['Kurve','Progress', 'Status', 'Fehlermeldung']
        self.window.iconbitmap(r"C:\Users\yangx\OneDrive - KUKA AG\EH\KI.ico")
        self.window.geometry("600x685")
        self.current_mode = 'Automatisch'
        self.server_socket = None
        self.selector = None
        self.running = False
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
        self.image_label.grid(row=0, column=0,columnspan=2,sticky="w",pady=9)
   
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
        self.motor = 'HV Test'
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
        self.fig.gca().set_xlabel('X')
        self.fig.gca().set_ylabel('Y')
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
        self.ProgName.grid(row=0, column=2, columnspan=2,pady=9,sticky="E")

        
        # 表头
        self.label_widgets = []
        
        for col, label_text in enumerate(self.labels, start=0):
            label = tk.Label(self.window, text=label_text, relief=tk.RIDGE, width=15)
            label.grid(row=4, column=col)
            self.label_widgets.append(label)
                
        # 数据初始化
        self.motor = 'Kurve'
        self.progress_values = {self.motor: 0}
        self.status_values = {self.motor: 'No Test'}
        self.error_values = {self.motor: ''}

        # 创建进度条、状态和错误消息的字典
        self.status_labels = {}
        self.error_labels = {}

        # 填充表格数据
        self.row = 5
        self.motor_label = tk.Label(self.window, text=self.motor)
        self.motor_label.grid(row=self.row, columnspan=1,column=0)

        # 状态
        self.status_labels[self.motor] = tk.Label(self.window, text=self.status_values[self.motor], width=15)
        self.status_labels[self.motor].grid(row=self.row,columnspan=1, column=2)

        # 错误消息
        self.error_labels[self.motor] = tk.Label(self.window, text=self.error_values[self.motor], width=15)
        self.error_labels[self.motor].grid(row=self.row,columnspan=1, column=3)

        #TCP
        self.Host_Label = tk.Label(self.window,text='SERVER HOST: ')
        self.Host_Entry =tk.Entry(self.window,width=15)
        self.Port_Label = tk.Label(self.window,text='SERVER PORT: ')
        self.Port_Entry =tk.Entry(self.window,width=15)
        self.Datenifo_text=tk.Text(self.window,width=45,height=1)       
        
        self.Host_Label.grid(row=1, column=0, pady=5,padx=2,columnspan=1,sticky=W)
        self.Host_Entry.grid(row=1, column=1, pady=5,padx=2,columnspan=2,sticky=W)
        self.Port_Label.grid(row=1, column=2, pady=5,padx=2,columnspan=1,sticky=E)
        self.Port_Entry.grid(row=1, column=3, pady=5,padx=2,columnspan=2,sticky=E)
        self.Datenifo_text.grid(row=2, column=1, pady=5,columnspan=3,sticky=E)

        self.connect_button = tk.Button(self.window, text="Start", command=lambda:Test_connect(self.connect_button, self.disconnect_button, self.Host_Entry, self.Port_Entry, self.Datenifo_text))
        self.disconnect_button = tk.Button(self.window, text="Stop", command=lambda:disconnect(self.connect_button, self.disconnect_button,self.Datenifo_text))
        self.connect_button.grid(row=2, column=0, pady=5,columnspan=1,sticky=W)
        self.disconnect_button.grid(row=2, column=0, pady=5,columnspan=1,sticky=E)      
       
        #Model auswählen
        self.Kern_button = Button(self.window, text="Select KI Modell", command=lambda: load_ai_model(self.Kern_button,self.KI_text))
        self.Kern_button.grid(row=3, column=3,pady=9,columnspan=1,sticky=E)
        #Test begin
        
        self.var = tk.IntVar()
        self.check_button = tk.Checkbutton(self.window, text="Test", variable=self.var, command=self.toggle_data_processing)
        self.check_button.grid(row=3, column=2, pady=9, columnspan=1, sticky=tk.E)


        self.KI_text = Text(self.window, height=2, width=45, wrap=WORD)
        self.KI_text.grid(row=3, column=0, pady=9, columnspan=3,sticky=W)



        self.Exit_button = Button(self.window, text="Quit", command=self.window.quit)
        self.Exit_button.grid(row=10, column=0, columnspan=4, pady=19)

        self.fig = plt.figure(figsize=(6, 4))
        #Jede Spalte 15
        self.fig.suptitle('Visualisierung der Testdaten')
        self.fig.gca().set_xlabel('X')
        self.fig.gca().set_ylabel('Y')
        # 将绘图对象添加到Tkinter窗口中
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=8, column=0,columnspan=4,pady=2)
    
    def toggle_data_processing(self):
        if self.var.get() == 1:
            
            self.start_data_processing()
        else:
            
            self.stop_data_processing()

    def classify_TCP(self,data):
        input_tensor = tensor(data.astype(np.float32))
        self.ai_model_path = r"{}".format(self.KI_text.get("1.0", "end-1c"))
        
        self.test_size = len(data)
        hidden_size =4 *self.test_size
        hidden_size2 =8 *self.test_size
        loaded_parameters = load(self.ai_model_path)
        loaded_model = NeuralNet(self.test_size, hidden_size,hidden_size2, 1)
        loaded_model.load_state_dict(loaded_parameters)
        start_time = time.time()
        output = loaded_model(input_tensor)

        end_time = time.time()
        predictions = (torch.sigmoid(output) >= 0.35).float()
        predicted_class = predictions.item()
        class_mapping = {1: "OK", 0: "NOT OK"}
        predicted_description = class_mapping.get(predicted_class, "Unknown")
        inference_time = end_time - start_time
        return inference_time, predicted_class, predicted_description, data
    
    def classify_wine(self):
        try:
            if self.sample_number < 2:
                    raise ValueError("Sample number is out of bounds.")
            
            self.csv_file_path = r"{}".format(self.csv_file_path)
            # CSV 文件路径
            data = pd.read_csv(self.csv_file_path)

            # 选择特定行的数据
            input_data_pandas = data.iloc[self.sample_number-2, 1:].values
            self.test_size = len(input_data_pandas)
            
            # 转换为 PyTorch tensor
            input_tensor = tensor(input_data_pandas.astype(np.float32))
             
            
            hidden_size =4 *self.test_size
            hidden_size2 =8 *self.test_size
            
            self.ai_model_path = r"{}".format(self.KI_text.get("1.0", "end-1c"))
            loaded_parameters = load(self.ai_model_path)
            loaded_model = NeuralNet(self.test_size, hidden_size,hidden_size2, 1)
            loaded_model.load_state_dict(loaded_parameters)
           
            start_time = time.time()
            output = loaded_model(input_tensor)
            print(input_tensor)
            print(output)
            out_sig=torch.sigmoid(output)
            print(torch.sigmoid(output))
            sig=torch.sigmoid(output).detach().numpy()
            print(torch.sigmoid(output).detach().numpy())
            print("Original output values (first 5):", out_sig[:5])
            print("Sigmoid output values (first 5):", sig[:5])
            end_time = time.time()
            predictions = (torch.sigmoid(output) >= 0.35).float()
            predicted_class = predictions.item()
            class_mapping = {1: "OK", 0: "NOT OK"}
            predicted_description = class_mapping.get(predicted_class, "Unknown")
            inference_time = end_time - start_time
            return inference_time, predicted_class, predicted_description, input_data_pandas

        except ValueError as ve:
            # 处理异常
            print("An exception occurred:", str(ve))
            return -1, -1, "Value Erro",0


        except Exception as e:
            print("An exception occurred:", str(e))
            return -1, -1,"Exception",0
     

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
        
        elif predicted_class == 1:
            self.s.theme_use('clam')
            self.s.configure("green.Horizontal.TProgressbar", foreground='green', background='green')

        else:
            self.s.theme_use('clam')
            self.s.configure("red.Horizontal.TProgressbar", foreground='red', background='red')   

        
    # 在循环之外创建进度条
        progress_bar_style = "yellow.Horizontal.TProgressbar" if predicted_class == -1 else \
                        "green.Horizontal.TProgressbar" if predicted_class == 1 else \
                        "red.Horizontal.TProgressbar"

        self.progress_bars[self.motor] = ttk.Progressbar(self.window, style=progress_bar_style, length=100)
        self.progress_bars[self.motor].grid(row=self.row, column=1)

        for i in range(101):
        # 更新进度条的值
            self.progress_bars[self.motor]['value'] = i
            self.window.update_idletasks()
        
        # 更新状态标签
            if predicted_class == 1:
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

    def Visualisierung(self,input_data_pandas):
        self.fig.clear()
        self.fig.suptitle('Visualisierung der Testdaten')
        max_value = 1.5*np.max(input_data_pandas)  # 使用 numpy.max() 获取数据中的最大值
        ax = self.fig.gca()
        ax.clear()  # 清空图形
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_ylim(0, max_value)  # 设置 y 轴范围
        ax.plot(input_data_pandas, color='blue')
        # 刷新画布
        self.canvas.draw_idle()
        self.canvas.flush_events()

    def on_generate_button_click(self):
        # 获取样本编号
        self.sample_number = int(self.sample_number_entry.get())

        # 调用 classify_wine 函数并获取返回值
        inference_time, predicted_class, predicted_description,input_data_pandas = self.classify_wine()
                                                                                                       
        self.Visualisierung(input_data_pandas)                                                                                    
        
        # 调用 simulate_data_loading 函数
        self.simulate_data_loading(predicted_class,inference_time)

        # 更新错误消息
        self.update_fehlermeldung(predicted_description)

    def start_data_processing(self):
        self.running = True
        self.process_data()

    def stop_data_processing(self):
        self.running = False
        self.Datenifo_text.delete("1.0", tk.END)
        self.Datenifo_text.insert(tk.END, "stopped")
  

    def process_data(self):
        global conn
        conn = server.conn
        self.window.after(1000, self.process_data)
        #print("Value of conn:", conn)    
        if not data_queue.empty():
            data = data_queue.get()
    # 调用 classify_TCP 函数并获取返回值
            inference_time, predicted_class, predicted_description, data = self.classify_TCP(data)
            self.Visualisierung(data)
            self.simulate_data_loading(predicted_class, inference_time)
            self.update_fehlermeldung(predicted_description)
            try:
                conn.send(predicted_description.encode())
                print("Sent predicted description to client:", predicted_description)
            except Exception as e:
                print("Failed to send predicted description to client:", e)            

        else:    
            # 继续检查数据，每100毫秒一次
            self.Datenifo_text.delete("1.0", tk.END)
            self.Datenifo_text.insert(tk.END, "Waiting for curve...") 


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
        if hasattr(self, 'progress_bars') and self.motor in self.progress_bars:
            self.progress_bars[self.motor].grid_forget()
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
        if hasattr(self, 'progress_bars') and self.motor in self.progress_bars:
            self.progress_bars[self.motor].grid_forget()
        self.image_label.grid_forget()
        self.ProgName.grid_forget()
        self.Host_Label.grid_forget()
        self.Host_Entry.grid_forget()
        self.Port_Label.grid_forget()
        self.Port_Entry.grid_forget()
        self.Datenifo_text.grid_forget()
        self.connect_button.grid_forget()
        self.disconnect_button.grid_forget()
        self.Kern_button.grid_forget()
        self.Exit_button.grid_forget()
        self.KI_text.grid_forget()
        self.canvas_widget.grid_forget()
        
        self.check_button.grid_forget()

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
            server_stop(self.server_socket, self.selector)
            self.destroy_Automatisch_widgets()
            self.create_Manuell_widgets()

        else: 
            self.current_mode = 'Manuell'         
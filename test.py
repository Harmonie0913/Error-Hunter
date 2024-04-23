import tkinter as tk
from tkinter import *
from tkinter import ttk
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from torch.utils.data import Dataset
import numpy as np
from matplotlib.figure import Figure
from tkinter import filedialog
import sys
from tkinter.scrolledtext import ScrolledText
# 设备配置
device = torch.device('cpu')

# 超参数

hidden_size1 = 20
num_classes = 2

batch_size = 4


def on_choose_traindata_button_click(csv_button, Trainsinfo_text):
    global train_size,train_loader
    # 获取用户选择的 CSV 文件路径
    csv_file_path = filedialog.askopenfilename(title="Select Traindata", filetypes=[("CSV Files", "*.csv")])

    # 检查是否选择了文件
    if csv_file_path:
        # change the color
        csv_button.configure(bg="green")
        Trainsinfo_text.delete(1.0, tk.END) 
        #Trainsinfo_text.insert(tk.END,f"Selected CSV file: {csv_file_path}")
        
        dataset1 = WineDataset(csv_file_path)
        train_loader = torch.utils.data.DataLoader(dataset=dataset1,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
        
        _,_,train_size = WineDataset.__getitem__(dataset1,1)
        ##这里 WineDataset 是一个类，所以我们需要传递一个实例作为第一个参数。
     
        Trainsinfo_text.insert(tk.END,f"{train_size} Merkmale\n")
        # Print class counts
        #print("Class Counts:")
        class_counts = dataset1.class_counts()
        
        # 定义一个字典来映射label到相应的名称
        label_mapping = {1: "IO", 2: "NIO"}

        # 输出映射后的名称
        for label, count in class_counts.items():
            if label in label_mapping:
                label_name = label_mapping[label]
            else:
                label_name = str(label)  # 如果label没有在映射中，则使用原始的label
            Trainsinfo_text.insert(tk.END, f"{label_name} Daten: {count} Proben\n")

    return train_loader, train_size


def on_choose_testdata_button_click(text,csv_button):
    global test_loader
    
    text.delete(1.0, tk.END)
    # 获取用户选择的 CSV 文件路径
    csv_file_path = filedialog.askopenfilename(title="Select Testdata", filetypes=[("CSV Files", "*.csv")])

    # 检查是否选择了文件
    if csv_file_path:
        # 将文件路径显示在界面上，或进行其他处理
        csv_button.configure(bg="green")
        #print(f"Selected CSV file: {csv_file_path}")
        dataset = WineDataset(csv_file_path)
        test_loader = torch.utils.data.DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
        
        _,_,sample_size = WineDataset.__getitem__(dataset,1)
        ##这里 WineDataset 是一个类，所以我们需要传递一个实例作为第一个参数。       
        text.insert(tk.END,f"{sample_size} Merkmale\n")
        #print(sample_size)
        # Print class counts
        #print("Class Counts:")
        class_counts = dataset.class_counts()
        # 定义一个字典来映射label到相应的名称
        label_mapping = {1: "IO", 2: "NIO"}

        # 输出映射后的名称
        for label, count in class_counts.items():
            if label in label_mapping:
                label_name = label_mapping[label]
            else:
                label_name = str(label)  # 如果label没有在映射中，则使用原始的label
            text.insert(tk.END, f"{label_name} Daten: {count} Proben\n")
    
    return test_loader,sample_size,label,count

class WineDataset(Dataset):

    def __init__(self,csv_file_path):
        # Initialize data, download, etc.
        # read with numpy or pandas
        xy = np.loadtxt(csv_file_path, delimiter=',', dtype=np.float32, skiprows=1) 
        self.n_samples = xy.shape[0]

        # here the first column is the class label, the rest are the features
        self.x_data = torch.from_numpy(xy[:, 1:]) # size [n_samples, n_features]
        self.y_data = torch.from_numpy(xy[:, 0]).to(torch.long)

        # Calculate and store sample sizes
        self.sample_sizes = [len(sample) for sample in self.x_data]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.sample_sizes[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
        # Function to get count of each class in the dataset
    def class_counts(self):
        class_count = {}
        for label in self.y_data.tolist():
            if label not in class_count:
                class_count[label] = 1
            else:
                class_count[label] += 1
        return class_count




class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, msg):
        self.text_widget.insert(tk.END, msg)
        self.text_widget.see(tk.END)  # 自动滚动到文本末尾

trained_model = None

# 创建一个函数来封装训练和可视化过程
def train_and_visualize(epoch_entry,rate_entry,ax_loss,ax_acc,canvas,text,ge_text):
    
    global train_loader,train_size,trained_model

    text.delete(1.0, tk.END)
    ge_text.delete(1.0, tk.END)
    # 初始化空列表以存储损失和准确度值
    input_size = train_size
    
    num_epochs = int(epoch_entry.get())
    learning_rate = float(rate_entry.get())

    # 创建模型

    print("Training model...\n")
    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size1, num_classes):
            super(NeuralNet, self).__init__()
            self.input_size = input_size
            self.l1 = nn.Linear(input_size, hidden_size1) 
            self.relu = nn.PReLU()
            self.l2 = nn.Linear(hidden_size1, num_classes)

        def forward(self, x):
            out = self.l1(x)
            out = self.relu(out)
            out = self.l2(out)
            return out

    model = NeuralNet(input_size, hidden_size1, num_classes).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    
    # 在循环之外定义损失和准确度列表

    losses = []
    accuracies = []
    accs = []
    #trained_model = train_and_visualize()
# 循环训练模型
    for epoch in range(num_epochs):
        running_correct = 0
        running_loss = 0.0

        #check the shape of the database

        # for i, data in enumerate(train_loader):
        #     print(i, data)


        for i, data in enumerate(train_loader):
            inputs, labels, sample_counts = data
            
            inputs = inputs
            labels = labels-1

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录损失值
            running_loss += loss.item()

            # 计算和记录准确度
            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                running_correct += correct
                accuracy = running_correct / ((i + 1) * batch_size) * 100

                # 更新损失曲线和准确度曲线
                losses.append(running_loss / (i + 1))  # 添加平均损失值
                accuracies.append(accuracy)  # 添加准确度值
                # Print the loss value and accuracy every step
                #print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / (i + 1):.4f}, Accuracy: {accuracy:.2f}%')
                # 清除子图内容并重新绘制曲线
                ax_loss.clear()
                ax_loss.plot(losses, color='blue')
                
                ax_loss.set_title('Training Loss')
                ax_loss.set_xlabel('Training Steps')
                ax_loss.set_ylabel('Loss')

                # ax_accuracy.clear()
                # ax_accuracy.plot(accuracies, color='red')
                # ax_accuracy.set_title('Training Accuracy')
                # ax_accuracy.set_xlabel('Training Steps')
                # ax_accuracy.set_ylabel('Accuracy (%)')

                # 刷新画布
                canvas.draw_idle()
                canvas.flush_events()

            with torch.no_grad():
                n_correct = 0
                n_samples = 0
                for i, data in enumerate(test_loader):
                    inputs, labels, sample_counts = data
                    labels = labels-1
                    outputs = model(inputs)
                    # max returns (value ,index)
                    _, predicted = torch.max(outputs.data, 1)
                    #predicted = predicted.unsqueeze(1)
                    #labels -= 1
                    #print(labels)
                    #print(predicted)

                    n_samples += labels.size(0)
                    n_correct += (predicted == labels).sum().item()
               
                acc = 100.0 * n_correct / n_samples       
                
                accs.append(acc)  # 添加准确度值

                ax_acc.clear()
                ax_acc.plot(accs, color='red')
                ax_acc.set_title('Validation Accuracy')
                ax_acc.set_xlabel('Training Steps')
                ax_acc.set_ylabel('Accuracy (%)')


    ge_text.insert(tk.END,f"{acc}%")        
    
    #print(f'Die Genauigkeit des akutuellen Modell ist:{acc}%')
        
 
    print("Die Lernrate und die Anzahl der Trainingsrunden können gemäß Standardwerten eingestellt werden.\nWenn die Genauigkeit des Modells nicht zufriedenstellend sind, kann versucht werden, das Training mehrmals zu wiederholen oder die Lernrate zu reduzieren und die Anzahl der Trainingsrunden zu erhöhen.")
    
    trained_model = model
    
    return     trained_model


# 定义保存模型的函数
def save_model( ):
    
    global trained_model 
    
    # 如果没有训练好的模型，则直接返回
    if trained_model is None:
        print("No trained model available.")
        return
    
    # 提示用户选择保存位置
    file_path = filedialog.asksaveasfilename(defaultextension=".pth", filetypes=[("Model files", "*.pth")])
    
    # 如果用户取消选择位置，则直接返回
    if not file_path:
        return
    
    # 保存模型
    torch.save(trained_model.state_dict(), file_path)
    print("Model saved successfully.")



def load_ai_model(kern_button,KI_text):
    ai_model_path = filedialog.askopenfilename(title="Select AI Model", filetypes=[("Model files", "*.pth")])
        # 检查是否选择了文件
    if ai_model_path:
        # 将文件路径显示在界面上，或进行其他处理
        kern_button.configure(bg="green")
        #print(f"Selected KI Modell: {self.ki_file_path}")
        KI_text.delete(1.0, END)  # 清空文本框
        KI_text.insert(END, ai_model_path)
    return ai_model_path

def entwickeln_and_visualize(epoch_entry, rate_entry, ax_loss, ax_acc, canvas, text, ge_text):
    global train_loader, train_size, entwickelt_model

    text.delete(1.0, tk.END)
    ge_text.delete(1.0, tk.END)

    input_size = train_size
    num_epochs = int(epoch_entry.get())
    learning_rate = float(rate_entry.get())

    # 获取加载后的模型路径
    ai_model_path = load_ai_model()
    if not ai_model_path:
        return None  # 如果未成功加载模型，则返回

    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size1, num_classes):
            super(NeuralNet, self).__init__()
            self.input_size = input_size
            self.l1 = nn.Linear(input_size, hidden_size1) 
            self.relu = nn.PReLU()
            self.l2 = nn.Linear(hidden_size1, num_classes)

        def forward(self, x):
            out = self.l1(x)
            out = self.relu(out)
            out = self.l2(out)
            return out

    # 创建模型
    model = NeuralNet(input_size, hidden_size1, num_classes).to(device)

    # 加载模型
    loaded_parameters = torch.load(ai_model_path)
    model.load_state_dict(loaded_parameters)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    accuracies = []
    accs = []

# 循环训练模型
    for epoch in range(num_epochs):
        running_correct = 0
        running_loss = 0.0

        #check the shape of the database

        # for i, data in enumerate(train_loader):
        #     print(i, data)


        for i, data in enumerate(train_loader):
            inputs, labels, sample_counts = data
            
            inputs = inputs
            labels = labels-1

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录损失值
            running_loss += loss.item()

            # 计算和记录准确度
            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                running_correct += correct
                accuracy = running_correct / ((i + 1) * batch_size) * 100

                # 更新损失曲线和准确度曲线
                losses.append(running_loss / (i + 1))  # 添加平均损失值
                accuracies.append(accuracy)  # 添加准确度值
                # 清除子图内容并重新绘制曲线
                ax_loss.clear()
                ax_loss.plot(losses, color='blue')
                
                ax_loss.set_title('Training Loss')
                ax_loss.set_xlabel('Training Steps')
                ax_loss.set_ylabel('Loss')


                # 刷新画布
                canvas.draw_idle()
                canvas.flush_events()

            with torch.no_grad():
                n_correct = 0
                n_samples = 0
                for i, data in enumerate(test_loader):
                    inputs, labels, sample_counts = data
                    labels = labels-1
                    outputs = model(inputs)
                    # max returns (value ,index)
                    _, predicted = torch.max(outputs.data, 1)

                    n_samples += labels.size(0)
                    n_correct += (predicted == labels).sum().item()
               
                acc = 100.0 * n_correct / n_samples       
                
                accs.append(acc)  # 添加准确度值

                ax_acc.clear()
                ax_acc.plot(accs, color='red')
                ax_acc.set_title('Validation Accuracy')
                ax_acc.set_xlabel('Training Steps')
                ax_acc.set_ylabel('Accuracy (%)')


    ge_text.insert(tk.END,f"{acc}%")        
    

    print("Die Lernrate und die Anzahl der Trainingsrunden können gemäß Standardwerten eingestellt werden.\nWenn die Genauigkeit des Modells nicht zufriedenstellend sind, kann versucht werden, das Training mehrmals zu wiederholen oder die Lernrate zu reduzieren und die Anzahl der Trainingsrunden zu erhöhen.")
    
    entwickelt_model = model
    
    return     entwickelt_model



# 运行Tkinter主事件循环
def main():
   
    # 创建Tkinter窗口
    root = tk.Tk()
    root.title("Training Visualization")
    root.geometry("700x800")
    # 创建一个 Figure 对象


    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(5, 8), gridspec_kw={'height_ratios': [1, 1]})

    # 在两个子图中绘制空的曲线
    ax_loss.set_title('Training Loss')
    ax_loss.set_xlabel('Training Steps')
    ax_loss.set_ylabel('Loss')

    ax_acc.set_title('Validation Accuracy')
    ax_acc.set_xlabel('Training Steps')
    ax_acc.set_ylabel('Accuracy (%)')

    plt.subplots_adjust(hspace=0.5)  # 调整子图之间的垂直间距

    # 创建 FigureCanvasTkAgg 对象，将 Figure 对象嵌入到 Tkinter 窗口中
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side="right",fill='both', anchor=None,pady=0)

    frame_top1 = tk.Frame(root)
    frame_top1.pack(side="top", fill="x", pady=10)

    csv_button = tk.Button(frame_top1, text="Trainingsdaten hochladen", command=lambda:on_choose_traindata_button_click(csv_button,text))
    csv_button.pack()

    frame_top2 = tk.Frame(root)
    frame_top2.pack(side="top", fill="x", pady=10)

    csv2_button = tk.Button(frame_top2, text="Testdaten hochladen", command=lambda: on_choose_testdata_button_click(text,csv2_button))
    csv2_button.pack()
 
    # Text für hyperparameter
    frame_top = tk.Frame(root)
    frame_top.pack(side="top", fill="x", pady=40)

    # 创建第一个标签并将其放置在框架中
    sample_number_label_1 = tk.Label(frame_top, text="Geben Sie bitte ")
    sample_number_label_1.pack(side="top")

    # 创建一个空的框架作为分隔
    separator_frame = tk.Frame(frame_top)
    separator_frame.pack(side="top")

    # 创建第二个标签并将其放置在框架中
    sample_number_label_2 = tk.Label(frame_top, text="die Hyperparameter ein!")
    sample_number_label_2.pack(side="top")

    # 创建一个框架来容纳标签和输入框
    frame = tk.Frame(root)
    frame.pack(side="top", fill="x", pady=4)

    # 创建标签并将其放置在框架中
    input_label = tk.Label(frame, text="input size:")
    input_label.pack(side="left")

    input_entry = tk.Entry(frame, width=5)
    input_entry.pack(side="left")

    input_label1 = tk.Label(frame, text="(sample size)")
    input_label1.pack(side="left")

    frame1 = tk.Frame(root)
    frame1.pack(side="top", fill="x", pady=4)

    rate_label = tk.Label(frame1, text="learning rate:")
    rate_label.pack(side="left")

    rate_entry = tk.Entry(frame1, width=10)
    rate_entry.pack(side="left")

    frame2 = tk.Frame(root)
    frame2.pack(side="top", fill="x", pady=4)

    epoch_label = tk.Label(frame2, text="number of epochs:")
    epoch_label.pack(side="left")

    epoch_entry = tk.Entry(frame2, width=10)
    epoch_entry.pack(side="left")
    

    # 创建一个按钮来启动训练和可视化过程
    frame3 = tk.Frame(root)
    frame3.pack(side="top", fill="x", pady=10)

    train_button = ttk.Button(frame3, text="Start Training", command=lambda:train_and_visualize(epoch_entry,rate_entry,ax_loss,ax_acc,canvas,text))
    train_button.pack(side="top")

    # 创建一个按钮来启动训练和可视化过程
    frame4 = tk.Frame(root)
    frame4.pack(side="top", fill="x", pady=10)

    # 创建保存模型的按钮
    save_button = tk.Button(frame4, text="Save Model", command=save_model)
    save_button.pack()


    frame5 = tk.Frame(root)
    frame5.pack(side="top", fill="x", pady=10)
    # 创建文本框
    text = tk.Text(frame5, height=10, width=20,state='normal',wrap='word')
    text.pack(side='top')

    #创建输出重定向器并重定向标准输出流
    stdout_redirector = StdoutRedirector(text)
    sys.stdout = stdout_redirector





    root.mainloop()




if __name__ == "__main__":
    main()

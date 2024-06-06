import socket
import selectors
import tkinter as tk
import threading
import types
import csv
import tkinter as tk
from tkinter import filedialog
import os
import  torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
from queue import Queue, Empty
# 全局变量
TCP_file = None
received_lines = 0
server_running = False
trained_model = None
conn = None  # 全局变量
io_count = 0
nio_count = 0

# 设备配置
device = torch.device('cpu')

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



import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

hidden_size1 = 20
num_classes = 2

batch_size = 4

def sotieren(info_text):
    global TCP_file, data_count, train_size, train_loader, validation_loader

    df = pd.read_csv(TCP_file)  # 读取 CSV 文件

    # X是特征，y是标签
    # 读取第一列到第data_count列（包括data_count），假设列名是数字字符串
    X = df.loc[:, [str(i) for i in range(1, data_count + 1)]]
    y = df['Label']  # 标签列

    # 检查每个类别的样本数量
    class_counts = y.value_counts()
    for cls, count in class_counts.items():
        if count < 2:
            info_text.insert(tk.END, f"类别 {cls} 的样本数量不足，只有 {count} 个样本。\n")
            # 可以选择移除该类别或者进行其他处理，例如过采样
            y = y[y != cls]
            X = X[y.index]
    
    # 重新检查每个类别的样本数量
    class_counts = y.value_counts()
    for cls, count in class_counts.items():
        if count < 2:
            info_text.insert(tk.END, f"类别 {cls} 仍然样本数量不足，请检查数据。\n")
            return

    # 使用StratifiedShuffleSplit进行分层抽样
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=41)

    for train_index, validation_index in sss.split(X, y):
        X_train, X_validation = X.iloc[train_index], X.iloc[validation_index]
        y_train, y_validation = y.iloc[train_index], y.iloc[validation_index]

    # 合并训练集和测试集
    train_data = pd.concat([y_train, X_train], axis=1)
    validation_data = pd.concat([y_validation, X_validation], axis=1)

    # 获取TCP文件所在的文件夹路径
    folder_path = os.path.dirname(TCP_file)

    # 新文件的路径
    validation_file = os.path.join(folder_path, os.path.splitext(os.path.basename(TCP_file))[0] + "_validation.csv")
    train_file = os.path.join(folder_path, os.path.splitext(os.path.basename(TCP_file))[0] + "_train.csv")

    # 保存为新的CSV文件
    train_data.to_csv(train_file, index=False, quoting=csv.QUOTE_NONE)
    validation_data.to_csv(validation_file, index=False, quoting=csv.QUOTE_NONE)

    # 加载数据集
    dataset1 = WineDataset(train_file)
    train_loader = torch.utils.data.DataLoader(dataset=dataset1,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0)

    _, _, train_size = WineDataset.__getitem__(dataset1, 1)

    dataset2 = WineDataset(validation_file)
    validation_loader = torch.utils.data.DataLoader(dataset=dataset2,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=0)

    _, _, validation_size = WineDataset.__getitem__(dataset2, 1)

    #info_text.insert(tk.END, f"{validation_size} Merkmale\n")
    # Print class counts
    #print("Class Counts:")
    class_count1 = dataset1.class_counts()
    class_count2 = dataset2.class_counts()

    # 定义一个字典来映射label到相应的名称
    label_mapping = {1: "IO", 2: "NIO"}

    # 输出映射后的名称
    info_text.insert(tk.END, f"Trainingsdatensatz\n")
    for label, count in class_count1.items():
        if label in label_mapping:
            label_name = label_mapping[label]
        else:
            label_name = str(label)  # 如果label没有在映射中，则使用原始的label
        info_text.insert(tk.END, f"{label_name} Daten: {count} Proben\n")

    info_text.insert(tk.END, f"Validierungsdatensatz\n")
    for label, count in class_count2.items():
        if label in label_mapping:
            label_name = label_mapping[label]
        else:
            label_name = str(label)  # 如果label没有在映射中，则使用原始的label
        info_text.insert(tk.END, f"{label_name} Daten: {count} Proben\n")

    return train_loader, train_size, validation_loader, validation_size







# 创建空的 CSV 文件
def create_TCP_csv(info_Datensatz):
    global TCP_file,data_count,received_lines,labels_written,io_count,nio_count
    received_lines = 0
    TCP_file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    labels_written = False
    info_Datensatz.delete("1.0", tk.END)
    info_Datensatz.insert(tk.END, f"Received {received_lines} curves.")
    io_count = 0
    nio_count = 0

def get_last_row_data(csv_file):
    last_type_no = ""
    last_test_no = ""
    with open(csv_file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if row:  # 跳过空行
                last_type_no = row[-2]
                last_test_no = row[-1]
    return last_type_no, last_test_no

def open_existing_csv(info_Datensatz, info_type, info_test):
    global TCP_file, received_lines, data_count, io_count, nio_count, labels_written

    # 选择 CSV 文件
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return

    TCP_file = file_path
    received_lines = 0
    data_count = 0
    io_count = 0
    nio_count = 0
    labels_written = False
    file_name = os.path.basename(TCP_file)

    # 读取最后一行数据的 type 和 test
    last_type_no, last_test_no = get_last_row_data(TCP_file)

    with open(TCP_file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        headers = next(csv_reader)
        data_count = len(headers) - 3
        labels_written = True

        for row in csv_reader:
            received_lines += 1
            if row[0] == "1":
                io_count += 1
            elif row[0] == "2":
                nio_count += 1
        received_lines = received_lines-1

    info_Datensatz.delete("1.0", tk.END)
    info_Datensatz.insert(tk.END, f"Loaded file {file_name}\n")
    info_Datensatz.insert(tk.END, f"Received {received_lines} curves.\n")  # 减去标签
    info_Datensatz.insert(tk.END, f"Merkmale:{data_count}\n")
    info_Datensatz.insert(tk.END, f"IO:{io_count} Proben\n")
    info_Datensatz.insert(tk.END, f"NIO:{nio_count} Proben\n")
    info_type.delete("1.0", tk.END)
    info_type.insert(tk.END, last_type_no)
    info_test.delete("1.0", tk.END)
    info_test.insert(tk.END, last_test_no)

def save_csv(info_Datensatz):
    global TCP_file

    # 选择保存文件的路径和名称
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return

    # 将当前的 TCP_file 复制到新的保存路径
    with open(TCP_file, 'r') as original_file:
        data = original_file.read()

    with open(file_path, 'w') as new_file:
        new_file.write(data)

    info_Datensatz.insert(tk.END,f"CSV file saved as {file_path}")


def start_server(server_socket, selector, info_Datensatz, info_type, info_test):
    global TCP_file
    global received_lines
    global data_count, labels_written
    global io_count, nio_count

    while server_running:
        events = selector.select()
        for key, mask in events:
            if key.fileobj == server_socket:
                conn, addr = server_socket.accept()
                conn.setblocking(False)
                data = types.SimpleNamespace(addr=addr, inb=b"", outb=b"")
                events = selectors.EVENT_READ | selectors.EVENT_WRITE
                selector.register(conn, events, data=data)
                info_Datensatz.delete("1.0", tk.END)
                info_Datensatz.insert(tk.END, f"Connection established with client: {addr}")
                info_type.insert(tk.END, "null")
                info_test.insert(tk.END, "null")

            else:
                conn = key.fileobj
                data = key.data
                try:
                    recv_data = conn.recv(10240)  # 尝试接收数据
                except BlockingIOError:
                    recv_data = b""

                if recv_data:

                    print("Received data from client:", recv_data.decode())  # 打印接收到的数据
                    data = recv_data.decode().rstrip(',')

                    # 分离 TypeNo 和 TestNo
                    *data_parts, type_no_str, test_no_str = data.split(',')

                    # 将前面的数据转换为NumPy数组
                    data_array = np.array(data_parts, dtype=float)

                    # 获取标签信息，假设标签在数据数组的第一个位置
                    label = int(data_array[0])

                    # 移除标签
                    data_without_label = data_array[1:]
                    data_count = len(data_without_label)

                    # 获取 TypeNo 和 TestNo
                    last_type_no = type_no_str
                    last_test_no = test_no_str

                    # 根据标签累加计数
                    if label == 1:
                        io_count += 1
                    elif label == 2:
                        nio_count += 1

                    # 检查是否已经写入了标签行，如果没有，则写入
                    if not labels_written:
                        with open(TCP_file, 'a', newline='') as csvfile:
                            csv_writer = csv.writer(csvfile)
                            # 要包括 data_count，你需要将 range(1, data_count + 1)。这样可以确保序列从 1 到 data_count，包括 data_count 本身。
                            labels = ["Label"] + [str(i) for i in range(1, data_count + 1)]+["Type"]+["Test"]
                            print(f"data_count:{data_count}\n")
                            csv_writer.writerow(labels)
                        labels_written = True

                    # 将数据写入 CSV 文件
                    with open(TCP_file, 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow(data.split(','))  # 将逗号分隔的数据拆分后写入 CSV 文件

                    # 更新文本框内容
                    received_lines += 1
                    info_Datensatz.delete("1.0", tk.END)
                    info_Datensatz.insert(tk.END, f"Received {received_lines} curves.\n")  # 减去标签
                    info_Datensatz.insert(tk.END, f"Merkmale:{data_count}\n")
                    info_Datensatz.insert(tk.END, f"IO:{io_count} Proben\n")
                    info_Datensatz.insert(tk.END, f"NIO:{nio_count} Proben\n")

                    # 在单独的文本框中显示 TypeNo 和 TestNo
                    info_type.delete("1.0", tk.END)
                    info_type.insert(tk.END, last_type_no)

                    info_test.delete("1.0", tk.END)
                    info_test.insert(tk.END, last_test_no)

                if recv_data == None:
                    info_Datensatz.delete("1.0", tk.END)
                    info_Datensatz.insert(tk.END, "Received no data" + "\n")




def connect(connect_button, disconnect_button, host_entry, port_entry, info_Datensatz,info_type,info_test):
    global server_socket, selector, server_running

    HOST = host_entry.get()
    PORT = int(port_entry.get())

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen()

    info_Datensatz.delete("1.0", tk.END)
    info_Datensatz.insert(tk.END, "Server is running. Waiting for connections...\n")

    selector = selectors.DefaultSelector()
    selector.register(server_socket, selectors.EVENT_READ, data=None)

    server_running = True

    connect_button.config(state=tk.DISABLED)
    disconnect_button.config(state=tk.NORMAL)

    start_server_thread = threading.Thread(target=start_server, args=(server_socket, selector, info_Datensatz,info_type,info_test))
    start_server_thread.start()

server_running = True
data_queue = Queue()

def Test_start_server(server_socket, selector, info_Datensatz):
    global conn
    while server_running:
        events = selector.select(timeout=1)  # 超时时间为1秒，防止无限等待
        for key, mask in events:
            if key.fileobj == server_socket:
                conn, addr = server_socket.accept()
                conn.setblocking(False)
                data = types.SimpleNamespace(addr=addr, inb=b"", outb=b"")
                events = selectors.EVENT_READ | selectors.EVENT_WRITE
                selector.register(conn, events, data=data)
                info_Datensatz.delete("1.0", tk.END)
                info_Datensatz.insert(tk.END, f"Connection established with client: {addr}")
            else:
                conn = key.fileobj
                data = key.data
                recv_data = b""
                try:
                    recv_data = conn.recv(10240)  # 尝试接收数据
                except BlockingIOError:
                    pass
                
                if recv_data:
                    print("Received data from client:", recv_data.decode())  # 打印接收到的数据
                    received_data = recv_data.decode().rstrip(',')
                    # 将接收到的字符串转换为NumPy数组
                    data_array = np.fromstring(received_data, sep=',')
                    data_queue.put(data_array)  # 将NumPy数组放入队列
                    data_count = len(data_array)
                    info_Datensatz.delete("1.0", tk.END)
                    info_Datensatz.insert(tk.END, f"Merkmale:{data_count}\n")  # 无标签
                    #conn.send(b"Hello client")
                    
                if recv_data == None:
                    info_Datensatz.delete("1.0", tk.END)
                    info_Datensatz.insert(tk.END, "Received no data\n")
    return  conn

def Test_connect(connect_button, disconnect_button, host_entry, port_entry, info_Datensatz):
    global server_socket, selector, server_running,conn

    HOST = host_entry.get()
    PORT = int(port_entry.get())

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen()

    info_Datensatz.delete("1.0", tk.END)
    info_Datensatz.insert(tk.END, "Server is running. Waiting for connections...\n")

    selector = selectors.DefaultSelector()
    selector.register(server_socket, selectors.EVENT_READ, data=None)

    server_running = True

    connect_button.config(state=tk.DISABLED)
    disconnect_button.config(state=tk.NORMAL)

    start_server_thread = threading.Thread(target=Test_start_server, args=(server_socket, selector, info_Datensatz))
    start_server_thread.daemon = True  # 设置为守护线程
    conn = start_server_thread.start()

    return conn





def disconnect(connect_button, disconnect_button, info_Datensatz):
    global server_socket, selector, server_running

    connect_button.config(state=tk.NORMAL)
    disconnect_button.config(state=tk.DISABLED)

    server_running = False

    server_socket.close()
    selector.close()

    info_Datensatz.delete("1.0", tk.END)
    info_Datensatz.insert(tk.END, "Server is stopped\n")




def server_stop(server_socket, selector):
    global server_running
    if  server_running == True:  # 设置标志为 False 来退出服务器循环        
        if server_socket is not None:
            server_socket.close()
        if selector is not None:
            selector.close()


def tcp_train_and_visualize(epoch_entry,rate_entry,ax_loss,ax_acc,canvas,text,ge_text):
    global trained_model
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
                for i, data in enumerate(validation_loader):
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


    ge_text.insert(tk.END,f"{accs[-1]}%")        
    
    #print(f'Die Genauigkeit des akutuellen Modell ist:{acc}%')
        
 
    print("Die Lernrate und die Anzahl der Trainingsrunden können gemäß Standardwerten eingestellt werden.\nWenn die Genauigkeit des Modells nicht zufriedenstellend sind, kann versucht werden, das Training mehrmals zu wiederholen oder die Lernrate zu reduzieren und die Anzahl der Trainingsrunden zu erhöhen.")
    
    trained_model = model
    
    return     trained_model




# 定义保存模型的函数
def tcp_save_model( ):
    
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
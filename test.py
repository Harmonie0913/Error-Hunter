import tkinter as tk
from tkinter import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from tkinter import filedialog


# 设备配置
device = torch.device('cuda')

# hyperparameter
trained_model = None
num_classes = 2
#batch_size = 2

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

class NeuralNet(nn.Module):
    def __init__(self, input_size,hidden_size1,hidden_size2,outsize):
        super(NeuralNet, self).__init__()
        hidden_size1 =     input_size
        hidden_size2 = 8 * input_size
        self.input_size = input_size
        outsize = 1
        self.l1 = nn.Linear(input_size, outsize) 
        # self.relu = nn.ReLU()
        # self.l2 = nn.Linear(hidden_size1,outsize)
        # self.relu2 = nn.LeakyReLU()
        # self.l3 = nn.Linear(hidden_size2, outsize)

    def forward(self, x):
        out = self.l1(x)
        # out = self.relu(out)
        # out = self.l2(out)
        # out = self.relu2(out)
        # out = self.l3(out)
        return out

def on_choose_traindata_button_click(batch_entry, csv_button, Trainsinfo_text):
    global train_size,train_loader
    # 获取用户选择的 CSV 文件路径
    csv_file_path = filedialog.askopenfilename(title="Select Traindata", filetypes=[("CSV Files", "*.csv")])
    batch_size = int(batch_entry.get())
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
        label_mapping = {1: "IO", 0: "NIO"}

        # 输出映射后的名称
        for label, count in class_counts.items():
            if label in label_mapping:
                label_name = label_mapping[label]
            else:
                label_name = str(label)  # 如果label没有在映射中，则使用原始的label
            Trainsinfo_text.insert(tk.END, f"{label_name} Daten: {count} Proben\n")

    return train_loader, train_size


def on_choose_testdata_button_click(batch_entry,text,csv_button):
   
    global validation_loader
    batch_size = int(batch_entry.get())
    text.delete(1.0, tk.END)
    # 获取用户选择的 CSV 文件路径
    csv_file_path = filedialog.askopenfilename(title="Select Testdata", filetypes=[("CSV Files", "*.csv")])

    # 检查是否选择了文件
    if csv_file_path:
        # 将文件路径显示在界面上，或进行其他处理
        csv_button.configure(bg="green")
        #print(f"Selected CSV file: {csv_file_path}")
        dataset = WineDataset(csv_file_path)
        validation_loader = torch.utils.data.DataLoader(dataset=dataset,
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
        label_mapping = {1: "IO", 0: "NIO"}

        # 输出映射后的名称
        for label, count in class_counts.items():
            if label in label_mapping:
                label_name = label_mapping[label]
            else:
                label_name = str(label)  # 如果label没有在映射中，则使用原始的label
            text.insert(tk.END, f"{label_name} Daten: {count} Proben\n")
    
    return validation_loader,sample_size,label,count


# 创建一个函数来封装训练和可视化过程
def train_and_visualize( epoch_entry, rate_entry, ax_loss, ax_acc, canvas, text, ge_text):
    global trained_model
    text.delete(1.0, tk.END)
    ge_text.delete(1.0, tk.END)
    
    input_size = train_size
    num_epochs = int(epoch_entry.get())
    learning_rate = float(rate_entry.get())
    
    # 创建模型
    text.insert(tk.END,"Training model...\n")

    hidden_size1 = 4 * input_size
    hidden_size2 = 8 * input_size
    outsize = 1

    model = NeuralNet(input_size,hidden_size1,hidden_size2,outsize).to(device)

    # 损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    
    # 在循环之外定义损失和准确度列表
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
   
    # 循环训练模型
    for epoch in range(num_epochs):
        model.train()
        running_correct = 0
        running_loss = 0.0
        total_samples = 0

        for i, data in enumerate(train_loader):
            inputs, labels, sample_sizes = data
            inputs = inputs.to(device)
            #将标签的形状从 [batch_size] 转换为 [batch_size, 1]
            labels = labels.unsqueeze(1).to(device)
            labels = labels.float()


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
            #nn.BCEWithLogitsLoss() 内部会自动应用 sigmoid 函数
            predictions = (torch.sigmoid(outputs) >= 0.5).float()  # 这里使用自定义阈值
            correct = (predictions == labels).sum().item()
            running_correct += correct
            total_samples += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100.0 * running_correct / total_samples

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # 验证模型
        model.eval()
        val_running_correct = 0
        val_running_loss = 0.0
        val_total_samples = 0

        with torch.no_grad():
            for i, data in enumerate(validation_loader):
                inputs, labels, sample_sizes = data
                inputs = inputs.to(device)
                labels = labels.unsqueeze(1).to(device)
                labels = labels.float()
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()

                predictions = (torch.sigmoid(outputs) >= 0.5).float()
                val_running_correct += (predictions == labels).sum().item()
                val_total_samples += labels.size(0)

        val_loss = val_running_loss / len(validation_loader)
        val_accuracy = 100.0 * val_running_correct / val_total_samples

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # 更新损失曲线和准确度曲线
        ax_loss.clear()
        ax_loss.plot(train_losses, label='Training Loss', color='blue')
        ax_loss.plot(val_losses, label='Validation Loss', color='red')
        ax_loss.set_title('Loss')
        ax_loss.set_xlabel('Epochs')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        
        ax_acc.clear()
        ax_acc.plot(train_accuracies, label='Training Accuracy', color='blue')
        ax_acc.plot(val_accuracies, label='Validation Accuracy', color='red')
        ax_acc.set_title('Accuracy')
        ax_acc.set_xlabel('Epochs')
        ax_acc.set_ylabel('Accuracy (%)')
        ax_acc.legend()

        # 刷新画布
        canvas.draw_idle()
        canvas.flush_events()

    ge_text.insert(tk.END, f"{val_accuracy}%")        

    text.insert(tk.END,"The learning rate can be set according to default values.\nIf the accuracy of the model is not satisfactory, you can try to repeat the training several times or reduce the learning rate and increase the number of training epochs.")

    trained_model = model

    return trained_model


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





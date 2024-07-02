import tkinter as tk
from tkinter import filedialog
import pandas as pd
import csv
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import os
from server import WineDataset

class SortierenWindow(tk.Toplevel):
    def __init__(self, window):
        self.window = tk.Toplevel(window)

        self.window.iconbitmap(r"C:\Users\yangx\OneDrive - KUKA AG\EH\KI.ico")
        self.window.title("Manuell Sortieren")
        self.train_loader = None
        self.validation_loader = None
        self.sample_size = 0
        self.validation_size = 0
        self.batch_size = 2  # 假设 batch_size 是 32
        self.csv_path = None


    def create_widgets(self):
        # Info über Datensatz und sortieren

        self.frame_Info = tk.Frame(self.window)
        self.frame_Info.pack(side="top", fill="x", pady=5)

        self.save_csv_button = tk.Button(self.frame_Info, text="Datensatz hochladen", command=lambda: self.on_choose_testdata_button_click(self.Info_Datensatz_text))
        self.Info_Datensatz_Label = tk.Label(self.frame_Info, text='Info über Datensatz')
        self.Info_Datensatz_text = tk.Text(self.frame_Info, height=7, width=50, wrap='word')
        self.sortieren_button = tk.Button(self.frame_Info, text="Sortieren", command=lambda: self.manuell_sortieren(self.Info_Datensatz_text))

        self.frame_Info.pack()
        self.Info_Datensatz_Label.pack()
        self.Info_Datensatz_text.pack()
        self.save_csv_button.pack(side="left")
        self.sortieren_button.pack(side="right")


    def on_choose_testdata_button_click(self, Info_Datensatz_text):
        self.csv_file_path = filedialog.askopenfilename(title="Select Testdata", filetypes=[("CSV Files", "*.csv")])
        
        if self.csv_file_path:
            self.save_csv_button.configure(bg="green")
            dataset = WineDataset(self.csv_file_path)
            self.validation_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                                 batch_size=self.batch_size,
                                                                 shuffle=True,
                                                                 num_workers=0)
            _, _, self.sample_size = dataset.__getitem__(1)
            Info_Datensatz_text.insert(tk.END, f"{self.sample_size} Merkmale\n")
            
            class_counts = dataset.class_counts()
            label_mapping = {1: "IO", 2: "NIO"}
            for label, count in class_counts.items():
                label_name = label_mapping.get(label, str(label))
                Info_Datensatz_text.insert(tk.END, f"{label_name} Daten: {count} Proben\n")


    # def on_choose_testdata_button_click(self, info_Datensatz):
    #     # 选择 CSV 文件
    #     self.csv_file_path = filedialog.askopenfilename(title="Select Testdata", filetypes=[("CSV Files", "*.csv")])
        
    #     if self.csv_file_path:
    #         self.save_csv_button.configure(bg="green")

    #         data_count = 0
    #         io_count = 0
    #         nio_count = 0
           
    #         file_name = os.path.basename(self.csv_file_path)

    #         with open(self.csv_file_path, 'r') as csvfile:
    #             csv_reader = csv.reader(csvfile)
    #             headers = next(csv_reader)
    #             data_count = len(headers) - 3

    #             for row in csv_reader:
    #                 if row[0] == "1":
    #                     io_count += 1
    #                 elif row[0] == "2":
    #                     nio_count += 1

    #         info_Datensatz.delete("1.0", tk.END)
    #         info_Datensatz.insert(tk.END, f"Loaded file {file_name}\n")

    #         info_Datensatz.insert(tk.END, f"Merkmale: {data_count}\n")
    #         info_Datensatz.insert(tk.END, f"IO: {io_count} Proben\n")
    #         info_Datensatz.insert(tk.END, f"NIO: {nio_count} Proben\n")



    
    
    def manuell_sortieren(self, info_text):

        df = pd.read_csv(self.csv_file_path)

        X = df.iloc[:,1:self.sample_size+1]
        y = df['Label']

        class_counts = y.value_counts()
        for cls, count in class_counts.items():
            if count < 2:
                info_text.insert(tk.END, f"类别 {cls} 的样本数量不足，只有 {count} 个样本。\n")
                y = y[y != cls]
                X = X.loc[y.index]
        
        class_counts = y.value_counts()
        for cls, count in class_counts.items():
            if count < 2:
                info_text.insert(tk.END, f"类别 {cls} 仍然样本数量不足，请检查数据。\n")
                return

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=41)
        for train_index, validation_index in sss.split(X, y):
            X_train, X_validation = X.iloc[train_index], X.iloc[validation_index]
            y_train, y_validation = y.iloc[train_index], y.iloc[validation_index]

        train_data = pd.concat([y_train, X_train], axis=1)
        validation_data = pd.concat([y_validation, X_validation], axis=1)

        folder_path = os.path.dirname(self.csv_file_path)
        validation_file = os.path.join(folder_path, os.path.splitext(os.path.basename(self.csv_file_path))[0] + "_validation.csv")
        train_file = os.path.join(folder_path, os.path.splitext(os.path.basename(self.csv_file_path))[0] + "_train.csv")

        train_data.to_csv(train_file, index=False, quoting=csv.QUOTE_NONE)
        validation_data.to_csv(validation_file, index=False, quoting=csv.QUOTE_NONE)

        dataset1 = WineDataset(train_file)
        self.train_loader = torch.utils.data.DataLoader(dataset=dataset1,
                                                        batch_size=self.batch_size,
                                                        shuffle=True,
                                                        num_workers=0)
        _, _, self.train_size = dataset1.__getitem__(1)

        dataset2 = WineDataset(validation_file)
        self.validation_loader = torch.utils.data.DataLoader(dataset=dataset2,
                                                             batch_size=self.batch_size,
                                                             shuffle=True,
                                                             num_workers=0)
        _, _, self.validation_size = dataset2.__getitem__(1)

        class_count1 = dataset1.class_counts()
        class_count2 = dataset2.class_counts()
        label_mapping = {1: "IO", 2: "NIO"}

        info_text.insert(tk.END, f"Trainingsdatensatz\n")
        for label, count in class_count1.items():
            label_name = label_mapping.get(label, str(label))
            info_text.insert(tk.END, f"{label_name} Daten: {count} Proben\n")

        info_text.insert(tk.END, f"Validierungsdatensatz\n")
        for label, count in class_count2.items():
            label_name = label_mapping.get(label, str(label))
            info_text.insert(tk.END, f"{label_name} Daten: {count} Proben\n")
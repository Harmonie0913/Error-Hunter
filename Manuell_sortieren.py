import tkinter as tk
from tkinter import filedialog
import pandas as pd
import csv
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import os
from server import KurvenDataset

class SortierenWindow(tk.Toplevel):
    def __init__(self, window):
        #Initialisierung der Fenstereigenschaften
        self.window = tk.Toplevel(window)
        self.window.iconbitmap(r"C:\Users\yangx\OneDrive - KUKA AG\EH\KI.ico")
        self.window.title("Manual split")
        self.train_loader = None
        self.validation_loader = None
        self.feauture_size = 0
        self.validation_size = 0
        self.batch_size = 2  #Das ist aber unabhängig von der für das Training verwendeten Batch Size, hier ist nur, um die Funktion des Dataloader zu nutzen 
        self.csv_path = None


    def create_widgets(self):
        # Info über Datensatz und sortieren

        self.frame_Info = tk.Frame(self.window)
        self.frame_Info.pack(side="top", fill="x", pady=5)

        self.save_csv_button = tk.Button(self.frame_Info, text="Upload the data set", command=lambda: self.on_choose_testdata_button_click(self.Info_Datensatz_text))
        self.Info_Datensatz_Label = tk.Label(self.frame_Info, text='Info about the data set')
        self.Info_Datensatz_text = tk.Text(self.frame_Info, height=7, width=50, wrap='word')
        self.sortieren_button = tk.Button(self.frame_Info, text="Split", command=lambda: self.manuell_sortieren(self.Info_Datensatz_text))

        self.frame_Info.pack()
        self.Info_Datensatz_Label.pack()
        self.Info_Datensatz_text.pack()
        self.save_csv_button.pack(side="left")
        self.sortieren_button.pack(side="right")


    def on_choose_testdata_button_click(self, Info_Datensatz_text):
        #Nur zur Anzeige von Informationen zum ausgewählten Datensatz
        self.csv_file_path = filedialog.askopenfilename(title="Select Test data", filetypes=[("CSV Files", "*.csv")])
        
        if self.csv_file_path:
            self.save_csv_button.configure(bg="green") # Buttonfarbe ändern, wenn Datei ausgewählt wurde
            dataset = KurvenDataset(self.csv_file_path)
            self.validation_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                                 batch_size=self.batch_size,
                                                                 shuffle=True,
                                                                 num_workers=0)
            _, _, self.feauture_size = dataset.__getitem__(1)
            Info_Datensatz_text.insert(tk.END, f"{self.feauture_size} Feautures\n")
            
            class_counts = dataset.class_counts()
            label_mapping = {1: "IO", 0: "NIO"}
            for label, count in class_counts.items():
                label_name = label_mapping.get(label, str(label))
                Info_Datensatz_text.insert(tk.END, f"{label_name} Data: {count} Samples\n")


    
    
    def manuell_sortieren(self, info_text):

        df = pd.read_csv(self.csv_file_path)
        #Trennung von Labels und Merkmalen
        X = df.iloc[:,1:self.feauture_size+1]
        y = df['Label']

        class_counts = y.value_counts()
        for cls, count in class_counts.items():
            if count < 2:
                info_text.insert(tk.END, f"The category {cls} has not enough samples, only {count} samples.\n")

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=41)#Der Datensatz wird in einem Verhältnis von acht zu zwei geteilt.
        for train_index, validation_index in sss.split(X, y):
            X_train, X_validation = X.iloc[train_index], X.iloc[validation_index]
            y_train, y_validation = y.iloc[train_index], y.iloc[validation_index]
        
        #einen neuen Trainingssatz und einen Validierungssatz erzeugt
        train_data = pd.concat([y_train, X_train], axis=1)
        validation_data = pd.concat([y_validation, X_validation], axis=1)

        folder_path = os.path.dirname(self.csv_file_path)
        validation_file = os.path.join(folder_path, os.path.splitext(os.path.basename(self.csv_file_path))[0] + "_validation.csv")
        train_file = os.path.join(folder_path, os.path.splitext(os.path.basename(self.csv_file_path))[0] + "_train.csv")

        train_data.to_csv(train_file, index=False, quoting=csv.QUOTE_NONE)
        validation_data.to_csv(validation_file, index=False, quoting=csv.QUOTE_NONE)
        
         #Nur zur Anzeige von Informationen zum neuen Trainingssatz und  Validierungssatz 
        dataset1 = KurvenDataset(train_file)
        self.train_loader = torch.utils.data.DataLoader(dataset=dataset1,
                                                        batch_size=self.batch_size,
                                                        shuffle=True,
                                                        num_workers=0)
        _, _, self.train_size = dataset1.__getitem__(1)

        dataset2 = KurvenDataset(validation_file)
        self.validation_loader = torch.utils.data.DataLoader(dataset=dataset2,
                                                             batch_size=self.batch_size,
                                                             shuffle=True,
                                                             num_workers=0)
        _, _, self.validation_size = dataset2.__getitem__(1)

        class_count1 = dataset1.class_counts()
        class_count2 = dataset2.class_counts()
        label_mapping = {1: "IO", 0: "NIO"}# Definiere ein Wörterbuch, um Bezeichnungen den entsprechenden Namen zuzuordnen.

        info_text.insert(tk.END, f"Training data set\n")
        for label, count in class_count1.items():
            label_name = label_mapping.get(label, str(label))# Ausgabe gemappter Namen
            info_text.insert(tk.END, f"{label_name} Data: {count} samples\n")# Wenn die Bezeichnung nicht in der Zuordnung enthalten ist, wird die ursprüngliche Bezeichnung verwendet.

        info_text.insert(tk.END, f"Validation data set\n")
        for label, count in class_count2.items():
            label_name = label_mapping.get(label, str(label))
            info_text.insert(tk.END, f"{label_name} Data: {count} samples\n")
import time
import tkinter as tk
from tkinter import *
import pandas as pd
import numpy as np
from PIL import Image, ImageTk
from tkinter import ttk,messagebox
import torch
from tkinter import filedialog
import time
import pandas as pd
import numpy as np
from torch import load, tensor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from test import load_ai_model
import server
from server import disconnect
from server import server_stop
from server import Test_connect
from server import data_queue
from test import NeuralNet
from Data_Preprocessor import DataPreprocessorWindow



class Mode2UIManager:
    def __init__(self, window):
        #Initialisierung der Fenstereigenschaften
        self.window = tk.Toplevel(window)
        self.labels = ['Curve', 'Progress', 'Status', 'Error message', 'Probability']
        self.window.iconbitmap(r"C:\Users\yangx\OneDrive - KUKA AG\EH\KI.ico")
        self.window.geometry("600x685")
        self.current_mode = 'Automatisch'
        self.server_socket = None
        self.selector = None
        self.running = False
        self.DataPreprocessor = None

    def insert_image(self,new_size):
        # KUKA Logo in der oberen linken Ecke einfügen
        image_path = r"C:\Users\yangx\OneDrive - KUKA AG\EH\KUKA_Logo.png" 
        original_image = Image.open(image_path)
        resized_image = original_image.resize(new_size)  
        photo = ImageTk.PhotoImage(resized_image)
        self.image_label = tk.Label(self.window)
        self.image_label.config(image=photo)
        self.image_label.image = photo  
        self.image_label.grid(row=0, column=0,columnspan=2,sticky="w",pady=9)

    def create_Data_Preprocessor(self):
        self.DataPreprocessor= DataPreprocessorWindow(self.window)
        self.DataPreprocessor.create_widgets()

    #Abkürzungen verwenden, um Messwerte kurven zu speichern 
    def save_plots(self, event):
        if event.key == 's':
            filename = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG files', '*.png')])
            if filename:
                self.ax.get_figure().savefig(filename)



    def create_Manuell_widgets(self):
        self.window.title("Test Mode")
        self.current_mode = 'Manuell'
        menubar = tk.Menu(self.window)
        mode_menu = tk.Menu(menubar, tearoff=0)
        mode_menu.add_command(label="Automatic", command=self.create_Automatisch_window)
        mode_menu.add_command(label="Manual", command=self.create_Manuell_window)
        menubar.add_cascade(label="Select Funktion", menu=mode_menu)
        menubar.add_cascade(label="Data Preprocessor", command=self.create_Data_Preprocessor)
        self.window.config(menu=menubar)
        
        
        new_size = (140, 24)
        self.insert_image(new_size)
        text = "Test Mode"
        font = ('times', 15, 'bold')
        color = "red"
        self.ProgName = tk.Label(self.window, text=text, font=font, fg=color)
        self.ProgName.grid(row=0, column=3, columnspan=2,pady=9,sticky="E")

        
        # Erstellen der Überschriften der Ergebnistabelle
        self.label_widgets = []
        
        for col, label_text in enumerate(self.labels, start=0):
            label = tk.Label(self.window, text=label_text, relief=tk.RIDGE, width=15)
            label.grid(row=4, column=col)
            self.label_widgets.append(label)
                
        # Initialisierung der Ergebnistabelle
        self.motor = 'HV Test'
        self.progress_values = {self.motor: 0}
        self.status_values = {self.motor: 'No Test'}
        self.error_values = {self.motor: ''}
        self.Wahrscheinlichkeit = {self.motor: ''}

        # Erstellen eines Wörterbuchs mit Status- und Fehlermeldungen
        self.status_labels = {}
        self.error_labels = {}
        self.Wahrscheinlichkeit_labels = {}
        
        self.row = 5
        self.motor_label = tk.Label(self.window, text=self.motor)
        self.motor_label.grid(row=self.row, column=0)

        # Status
        self.status_labels[self.motor] = tk.Label(self.window, text=self.status_values[self.motor], width=15)
        self.status_labels[self.motor].grid(row=self.row, column=2)

        # Fehermeldung
        self.error_labels[self.motor] = tk.Label(self.window, text=self.error_values[self.motor], width=15)
        self.error_labels[self.motor].grid(row=self.row, column=3)
        
        # Wahrschenlichkeit
        self.Wahrscheinlichkeit_labels[self.motor] = tk.Label(self.window, text=self.Wahrscheinlichkeit[self.motor], width=15)
        self.Wahrscheinlichkeit_labels[self.motor].grid(row=self.row, column=4)

        #csv auswählen
        self.generate_button1 = Button(self.window, text="Select Data", command=lambda: self.on_choose_csv_button_click(self.generate_button1))
        self.generate_button1.grid(row=1,column=4,pady=9,sticky=E)


        self.text_Field = Text(self.window, height=2, width=45, wrap=WORD)
        self.text_Field.grid(row=1, column=0, pady=9, columnspan=3,sticky=W)


        # Trainiertes KI Modell hochladen
        self.Kern_button = Button(self.window, text="Select AI Model", command=lambda: load_ai_model(self.Kern_button,self.KI_text))
        self.Kern_button.grid(row=2, column=4,pady=9,sticky=E)


        self.KI_text = Text(self.window, height=2, width=45, wrap=WORD)
        self.KI_text.grid(row=2, column=0, pady=9, columnspan=3,sticky=W)


        # Start Fehlererkennung
        self.generate_button2 = Button(self.window, text="CHECK IT NOW!", command=self.on_generate_button_click)
        self.generate_button2.grid(row=3, column=4,pady=9,sticky=E)


        # Eingabe der Probenummer
        self.sample_number_label = Label(self.window, text="Please enter your sample number:")
        self.sample_number_label.grid(row=3, column=0,pady=5,columnspan=2,sticky=W)
        
        self.sample_number_entry = tk.Entry(self.window,width=10)
        self.sample_number_entry.grid(row=3, column=3,pady=5,sticky=W)

        self.Exit_button = Button(self.window, text="Quit", command=self.window.quit)
        self.Exit_button.grid(row=9, column=2,pady=9)

        #Messwerte Kurvendarstellung, am Anfangen ist leer
        self.fig = plt.figure(figsize=(6, 4))
        self.fig.suptitle('Visualization of the test data')
        self.fig.gca().set_xlabel('X')
        self.fig.gca().set_ylabel('Y')

        # Hinzufügen von Zeichenobjekten zum Tkinter-Fenster
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=8, column=0,columnspan=5,pady=2)
        
        self.fig.canvas.mpl_connect('key_press_event', self.save_plots)

    def create_Automatisch_widgets(self):
        self.window.title("Test Mode")
        self.current_mode = 'Automatisch'
        menubar = tk.Menu(self.window)
        mode_menu = tk.Menu(menubar, tearoff=0)
        mode_menu.add_command(label="Automatic", command=self.create_Automatisch_window)
        mode_menu.add_command(label="Manual", command=self.create_Manuell_window)
        menubar.add_cascade(label="Select Function", menu=mode_menu)
        menubar.add_cascade(label="Data Preprocessor", command=self.create_Data_Preprocessor)
        self.window.config(menu=menubar)
        
        
        new_size = (140, 24)
        self.insert_image(new_size)
        text = "Test Mode"
        font = ('times', 15, 'bold')
        color = "red"
        self.ProgName = tk.Label(self.window, text=text, font=font, fg=color)
        self.ProgName.grid(row=0, column=3, columnspan=2,pady=9,sticky="E")

        
        # Erstellen der Überschriften der Ergebnistabelle
        self.label_widgets = []
        
        for col, label_text in enumerate(self.labels, start=0):
            label = tk.Label(self.window, text=label_text, relief=tk.RIDGE, width=15)
            label.grid(row=4, column=col)
            self.label_widgets.append(label)
                
       # Initialisierung der Ergebnistabelle
        self.motor = 'HV Test'
        self.progress_values = {self.motor: 0}
        self.status_values = {self.motor: 'No Test'}
        self.error_values = {self.motor: ''}
        self.Wahrscheinlichkeit = {self.motor: ''}

       # Erstellen eines Wörterbuchs mit Status- und Fehlermeldungen
        self.status_labels = {}
        self.error_labels = {}
        self.Wahrscheinlichkeit_labels = {}
        
        self.row = 5
        self.motor_label = tk.Label(self.window, text=self.motor)
        self.motor_label.grid(row=self.row, columnspan=1,column=0)

        # Status
        self.status_labels[self.motor] = tk.Label(self.window, text=self.status_values[self.motor], width=15)
        self.status_labels[self.motor].grid(row=self.row,columnspan=1, column=2)

        # Fehlermeldung
        self.error_labels[self.motor] = tk.Label(self.window, text=self.error_values[self.motor], width=15)
        self.error_labels[self.motor].grid(row=self.row,columnspan=1, column=3)

        # Wahrschenlichkeit
        self.Wahrscheinlichkeit_labels[self.motor] = tk.Label(self.window, text=self.Wahrscheinlichkeit[self.motor], width=15)
        self.Wahrscheinlichkeit_labels[self.motor].grid(row=self.row, column=4)

        #TCP Konfiguration
        self.Host_Label = tk.Label(self.window,text='SERVER HOST: ')
        self.Host_Entry =tk.Entry(self.window,width=15)
        self.Port_Label = tk.Label(self.window,text='SERVER PORT: ')
        self.Port_Entry =tk.Entry(self.window,width=15)
        self.Datenifo_text=tk.Text(self.window,width=45,height=1)       
        
        self.Host_Label.grid(row=1, column=0, pady=5,padx=2,columnspan=1,sticky=W)
        self.Host_Entry.grid(row=1, column=1, pady=5,padx=2,columnspan=1,sticky=E)
        self.Port_Label.grid(row=1, column=3, pady=5,padx=2,columnspan=1,sticky=W)
        self.Port_Entry.grid(row=1, column=4, pady=5,padx=2,columnspan=1,sticky=E)
        self.Datenifo_text.grid(row=2, column=2, pady=5,columnspan=3,sticky=E)

        self.connect_button = tk.Button(self.window, text="Start", command=lambda:Test_connect(self.connect_button, self.disconnect_button, self.Host_Entry, self.Port_Entry, self.Datenifo_text))
        self.disconnect_button = tk.Button(self.window, text="Stop", command=lambda:disconnect(self.connect_button, self.disconnect_button,self.Datenifo_text))
        self.connect_button.grid(row=2, column=0, pady=5,columnspan=1,sticky=W)
        self.disconnect_button.grid(row=2, column=1, pady=5,columnspan=1,sticky=W)      
       
        #Model auswählen
        self.Kern_button = Button(self.window, text="Select AI Modell", command=lambda: load_ai_model(self.Kern_button,self.KI_text))
        self.Kern_button.grid(row=3, column=0,pady=9,columnspan=1,sticky=W)
        #Test begin
        
        self.var = tk.IntVar()
        self.check_button = tk.Checkbutton(self.window, text="Test", variable=self.var, command=self.toggle_data_processing)
        self.check_button.grid(row=3, column=1, pady=9, columnspan=1, sticky=W)


        self.KI_text = Text(self.window, height=2, width=45, wrap=WORD)
        self.KI_text.grid(row=3, column=2, pady=9, columnspan=3,sticky=W)



        self.Exit_button = Button(self.window, text="Quit", command=self.window.quit)
        self.Exit_button.grid(row=9, column=2, pady=9)

        self.fig = plt.figure(figsize=(6, 4))
        #Jede Spalte 15
        self.fig.suptitle('Visualization of the test data')
        self.fig.gca().set_xlabel('X')
        self.fig.gca().set_ylabel('Y')
        # Hinzufügen von Zeichenobjekten zum Tkinter-Fenster
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=8, column=0,columnspan=5,pady=2)
    
    def toggle_data_processing(self):
        #Erfasst den Zustand des Checkbuttons, und wenn er 1 ist, d.h. der Checkbutton angekreuzt ist, wird die Funktion KI fehlererkenung aktiviert.
        if self.var.get() == 1:
            
            self.start_data_processing()
        else:
            
            self.stop_data_processing()

    def classify_TCP(self,data):
        #Feststellen, ob DataPreprocessor hochgeladen wurde
        if self.DataPreprocessor is None:
            messagebox.showinfo("Info","No preprocessor has been loaded.")

        input_tensor = tensor(self.DataPreprocessor.preprocessor.process(data).astype(np.float32))# Die über TCP übertragene Messwerte
        
        #Laden von ausgewähltem KI Modell 
        self.ai_model_path = r"{}".format(self.KI_text.get("1.0", "end-1c"))        
        self.test_size = len(input_tensor)
        hidden_size =4 *self.test_size
        hidden_size2 =8 *self.test_size
        loaded_parameters = load(self.ai_model_path)
        #Der NeuralNet-Aufruf aus test.py stellt sicher, dass das hier geladene KI-Modell die gleiche Struktur hat wie das zum Training verwendete Modell
        loaded_model = NeuralNet(self.test_size, hidden_size,hidden_size2, 1)
        loaded_model.load_state_dict(loaded_parameters)
        start_time = time.time()
        #Abrufen der Ergebnisse der Vorwärtspropagation zur Berechnung der Wahrscheinlichkeiten.
        output = loaded_model(input_tensor)
        out_sig=torch.sigmoid(output)
        value = round(out_sig.item(), 4)

        end_time = time.time()
        predictions = (torch.sigmoid(output) >= 0.5).float()
        predicted_class = predictions.item()
        class_mapping = {1: "OK", 0: "NOT OK"}
        predicted_description = class_mapping.get(predicted_class, "Unknown")# Verschiedene Kategorien definieren, Undefinierte Kategorien werden als "Unknown" angezeigt.
        inference_time = end_time - start_time#Berechnung der Bearbeitungszeit, Visualisierung der Bearbeitung als Fortschrittsbalken
        return inference_time, predicted_class, predicted_description, data,value
    
    def classify(self):
        try:
            if self.sample_number < 2:
                    raise ValueError("Sample number is out of bounds.")#Da die Daten alle in der zweiten Zeile der csv-Datei beginnen
            
            self.csv_file_path = r"{}".format(self.csv_file_path)
            # Pfad der CSV-Datei
            data = pd.read_csv(self.csv_file_path)

            # Wählen eine bestimmte Datenzeile aus.
            # Die letzten beiden Zeilen sind Typennummer und Testnummer, die nichts mit den Messdaten zu tun haben.
            input_data_pandas = data.iloc[self.sample_number-2, 1:-2].values
        
            #Feststellen, ob DataPreprocessor hochgeladen wurde
            if self.DataPreprocessor is None:
                messagebox.showinfo("Info","No preprocessor has been loaded.")

            # Umwandlung in PyTorch-Tensor
            input_tensor = tensor(self.DataPreprocessor.preprocessor.process(input_data_pandas).astype(np.float32)) 
            

            #Gleich wie in Classify_TCP
            self.test_size = len(input_tensor)                
            hidden_size =4 *self.test_size
            hidden_size2 =8 *self.test_size
            
            self.ai_model_path = r"{}".format(self.KI_text.get("1.0", "end-1c"))
            loaded_parameters = load(self.ai_model_path)
            loaded_model = NeuralNet(self.test_size, hidden_size,hidden_size2, 1)
            loaded_model.load_state_dict(loaded_parameters)           
            start_time = time.time()

            output = loaded_model(input_tensor)
            out_sig=torch.sigmoid(output)
        
            value = round(out_sig.item(), 4)
            end_time = time.time()

            predictions = (torch.sigmoid(output) >= 0.5).float()
            predicted_class = predictions.item()
            class_mapping = {1: "OK", 0: "NOT OK"}
            predicted_description = class_mapping.get(predicted_class, "Unknown")
            inference_time = end_time - start_time
            return inference_time, predicted_class, predicted_description, input_data_pandas,value

        except ValueError as ve:
            # Behandlung von Ausnahmen
            print("An exception occurred:", str(ve))
            return -1, -1, "Value Erro",0,0


        except Exception as e:
            print("An exception occurred:", str(e))
            return -1, -1,"Exception",0,0
     

    def simulate_data_loading(self,predicted_class,inference_time):
  
        # Prüfen, ob ein alter Fortschrittsbalken vorhanden ist, und er wird entfernt, falls er existiert.
        if hasattr(self, 'progress_bars') and self.motor in self.progress_bars:
            self.progress_bars[self.motor].grid_forget()
            

        self.progress_bars = {}
        self.s = ttk.Style()  # ttk.Style()-Objekt vor der Schleife erstellen

        if predicted_class == -1:#Wenn eine Ausnahme aufgetreten ist, wird der Fortschrittsbalken gelb.
            self.s.theme_use('clam')
            self.s.configure("yellow.Horizontal.TProgressbar", foreground='yellow', background='yellow')
        
        elif predicted_class == 1:#Wenn eine IO ist, wird der Fortschrittsbalken grün.
            self.s.theme_use('clam')
            self.s.configure("green.Horizontal.TProgressbar", foreground='green', background='green')

        else:
            self.s.theme_use('clam')#Wenn eine Andere(IO) ist, wird der Fortschrittsbalken rot.
            self.s.configure("red.Horizontal.TProgressbar", foreground='red', background='red')   

        
    # einen Fortschrittsbalken außerhalb der Schleife erstellen
        progress_bar_style = "yellow.Horizontal.TProgressbar" if predicted_class == -1 else \
                        "green.Horizontal.TProgressbar" if predicted_class == 1 else \
                        "red.Horizontal.TProgressbar"

        self.progress_bars[self.motor] = ttk.Progressbar(self.window, style=progress_bar_style, length=100)
        self.progress_bars[self.motor].grid(row=self.row, column=1)

        for i in range(101):
        # Aktualisierung des Fortschrittsbalkens
            self.progress_bars[self.motor]['value'] = i
            self.window.update_idletasks()
        
        # Aktualisierung des Status
            if predicted_class == 1:
                self.status_labels[self.motor]['text'] = "Processing..."
            else:
                self.status_labels[self.motor]['text'] = "WARNING!!!"
        
        # Berechnung der Bearbeitungszeit bei Erhalt des NIO.
            if predicted_class == -1:
                time.sleep(1e-20)
            else:
                time.sleep(inference_time / 100)


    # Aktualisierung der Fehlermeldung
    def update_fehlermeldung(self,predicted_description):
        self.error_labels[self.motor]['text'] = f"{predicted_description}"

    # Aktualisierung der Wahrscheinlichkeit
    def update_Wahrscheinlichkeit(self,value):
        self.Wahrscheinlichkeit_labels[self.motor]['text'] = f"{100 * value:.2f}%"

    def Visualisierung(self,input_data_pandas):
        self.fig.clear()
        self.fig.suptitle('Visualization of the test data')
        max_value = 1.5*np.max(input_data_pandas)   # Mit numpy.max() den Maximalwert in den Daten ermitteln

        self.ax = self.fig.gca()
        self.ax.clear()  # clear all
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_ylim(0, max_value) # Den Bereich der y-Achse festlegen
        self.ax.plot(input_data_pandas, color='blue')
        # Akutualisierung des Zeichens
        self.canvas.draw_idle()
        self.canvas.flush_events()

    def on_generate_button_click(self):
        # Probennummer ermitteln
        self.sample_number = int(self.sample_number_entry.get())

        # classify Anrufen 
        inference_time, predicted_class, predicted_description,input_data_pandas,value = self.classify()
                                                                                                       
        self.Visualisierung(input_data_pandas)                                                                                    
        
        #Aktualisierung des Fortschrittsbalkens
        self.simulate_data_loading(predicted_class,inference_time)

        # Aktualisierung der Fehlermeldung
        self.update_fehlermeldung(predicted_description)
        #Aktualisierung der Wahrscheinlichkeit
        self.update_Wahrscheinlichkeit(value)

    def start_data_processing(self):
        self.running = True
        self.process_data()

    def stop_data_processing(self):
        self.running = False
        self.Datenifo_text.delete("1.0", tk.END)
        self.Datenifo_text.insert(tk.END, "stopped")
  

    def process_data(self):
        #Verwendet eine TCP-Verbindung und wartet ständig auf den Empfang neuer Daten
        global conn
        conn = server.conn
        self.window.after(1000, self.process_data)#Aufruf der Methode self.process_data nach 1 Sekunde (1000 Millisekunden)
  
        if not data_queue.empty():
            data = data_queue.get()
    # 调用 classify_TCP 函数并获取返回值
            inference_time, predicted_class, predicted_description, data, value = self.classify_TCP(data)
            self.Visualisierung(data)
            self.simulate_data_loading(predicted_class, inference_time)
            self.update_fehlermeldung(predicted_description)
            self.update_Wahrscheinlichkeit(value)
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
        self.Wahrscheinlichkeit_labels[self.motor].grid_forget()
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
        self.Wahrscheinlichkeit_labels[self.motor].grid_forget()
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
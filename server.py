#TCP
import socket
import selectors
import threading
import types
from queue import Queue
#GUI
import tkinter as tk
from tkinter import filedialog
#KI
import  torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
#Datenanalyse
import csv
import os
import numpy as np
import pandas as pd
#Stellen die sicher, dass  in projekt die Form des verwendeten Datensatzes und die Form des KI-Modells immer geleich
from test import KurvenDataset
from test import NeuralNet

# Global Variante
TCP_file = None
received_lines = 0
server_running = False
trained_model = None
conn = None  
io_count = 0
nio_count = 0
num_classes = 2

# Gerätekonfigurationen für das KI-Training
#cuda für GPU
#CPU für CPU
#device = torch.device('cpu')
device = torch.device('cuda')

#Split Funktion
def sotieren(batch_entry,info_text):
    global TCP_file, data_count, train_size, train_loader, validation_loader
    
    batch_size = int(batch_entry.get())

    df = pd.read_csv(TCP_file)  

    # x ist das Merkmal, y ist das Label.
    # Lesen der ersten Spalte bis einschließlich data_count, vorausgesetzt, die Spaltennamen sind numerische Strings.
    if data_count == 0:
        X = df.loc[:, [str(i) for i in range(1, train_size + 1)]]
    else:
         X = df.loc[:, [str(i) for i in range(1, data_count + 1)]]
    

    y = df['Label']  

   # Anzahl der überprüften Proben für jede Kategorie
    class_counts = y.value_counts()
    for cls, count in class_counts.items():
        if count < 2:
            info_text.insert(tk.END, f"The category {cls} has not enough samples, only {count} samples.\n")

    #Der Datensatz wird in einem Verhältnis von acht zu zwei geteilt.
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=41)

    for train_index, validation_index in sss.split(X, y):
        X_train, X_validation = X.iloc[train_index], X.iloc[validation_index]
        y_train, y_validation = y.iloc[train_index], y.iloc[validation_index]

    #einen neuen Trainingssatz und einen Validierungssatz erzeugt
    train_data = pd.concat([y_train, X_train], axis=1)
    validation_data = pd.concat([y_validation, X_validation], axis=1)

    # Ermitteln des Pfads zu dem Ordner, in dem sich die TCP-file befindet
    folder_path = os.path.dirname(TCP_file)

    #Neuer Trainings- und Validierungssatz und TCP-flie werden in der gleichen Order gespeichert.
    validation_file = os.path.join(folder_path, os.path.splitext(os.path.basename(TCP_file))[0] + "_validation.csv")
    train_file = os.path.join(folder_path, os.path.splitext(os.path.basename(TCP_file))[0] + "_train.csv")

    train_data.to_csv(train_file, index=False, quoting=csv.QUOTE_NONE)
    validation_data.to_csv(validation_file, index=False, quoting=csv.QUOTE_NONE)

    # Laden Datensätze
    dataset1 = KurvenDataset(train_file)
    train_loader = torch.utils.data.DataLoader(dataset=dataset1,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0)

    _, _, train_size = KurvenDataset.__getitem__(dataset1, 1)

    dataset2 = KurvenDataset(validation_file)
    validation_loader = torch.utils.data.DataLoader(dataset=dataset2,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=0)

    _, _, validation_size = KurvenDataset.__getitem__(dataset2, 1)


    class_count1 = dataset1.class_counts()
    class_count2 = dataset2.class_counts()

    # Definiere ein Wörterbuch, um Bezeichnungen den entsprechenden Namen zuzuordnen.
    label_mapping = {1: "IO", 0: "NIO"}

    # Ausgabe gemappter Namen
    info_text.insert(tk.END, f"Training data set\n")
    for label, count in class_count1.items():
        if label in label_mapping:
            label_name = label_mapping[label]
        else:
            label_name = str(label)  # Wenn die Bezeichnung nicht in der Zuordnung enthalten ist, wird die ursprüngliche Bezeichnung verwendet.
        info_text.insert(tk.END, f"{label_name} Data: {count} samples\n")

    info_text.insert(tk.END, f"Validation data set\n")
    for label, count in class_count2.items():
        if label in label_mapping:
            label_name = label_mapping[label]
        else:
            label_name = str(label)  
        info_text.insert(tk.END, f"{label_name} Data: {count} samples\n")

    return train_loader, train_size, validation_loader, validation_size



# TCP file erzeugen
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
    #Ermittelt den Type Nummer und den Test Nummer der letzten Datenzeile, die verwendet wird, 
    #um die Informationen der letzten Kurve nach dem Hochladen des Datensatzs anzuzeigen, damit die Datendatei weiter aufgebaut werden kann.
    last_type_no = ""
    last_test_no = ""
    with open(csv_file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if row:  # Leerzeilen überspringen
                last_type_no = row[-2]
                last_test_no = row[-1]
    return last_type_no, last_test_no

#Hochladen des Datensatzes
def open_existing_csv(info_Datensatz, info_type, info_test):
    global TCP_file, received_lines, data_count, io_count, nio_count, labels_written

    # Datensatz auswählen
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

    # Ermittelt den Type Nummer und den Test Nummer der letzten Datenzeile
    last_type_no, last_test_no = get_last_row_data(TCP_file)

    with open(TCP_file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        headers = next(csv_reader)
        data_count = len(headers) - 3 #Type Nummer und Test Number und Label sind ausgeschlossen
        labels_written = True #Die Überschriften sind schon geschrieben
        
        #Zählen die Anzahl jeder Klassen
        for row in csv_reader:
            received_lines += 1
            if row[0] == "1":
                io_count += 1
            elif row[0] == "0":
                nio_count += 1
        received_lines = received_lines
    
    #Info auf GUI darstellen
    info_Datensatz.delete("1.0", tk.END)
    info_Datensatz.insert(tk.END, f"Loaded file {file_name}\n")
    info_Datensatz.insert(tk.END, f"Received {received_lines} curves.\n")  
    info_Datensatz.insert(tk.END, f"Feautures:{data_count}\n")
    info_Datensatz.insert(tk.END, f"IO:{io_count} samples\n")
    info_Datensatz.insert(tk.END, f"NIO:{nio_count} samples\n")
    info_type.delete("1.0", tk.END)
    info_type.insert(tk.END, last_type_no)
    info_test.delete("1.0", tk.END)
    info_test.insert(tk.END, last_test_no)

def save_csv(info_Datensatz):
    #um Verloren der Daten zu vermeiden
    global TCP_file

    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return

    # Kopieren der aktuellen TCP_file in den neuen Speicherpfad
    with open(TCP_file, 'r') as original_file:
        data = original_file.read()

    with open(file_path, 'w') as new_file:
        new_file.write(data)

    info_Datensatz.insert(tk.END,f"CSV file saved as {file_path}")


def start_server(server_socket, selector, info_Datensatz, info_type, info_test,info_tcpdata):
    global TCP_file
    global received_lines
    global data_count, labels_written
    global io_count, nio_count

    while server_running:#Dies ist eine Endlosschleife, und der Server läuft so lange, bis die Variable server_running auf False
        #Dies ist ein blockierender Aufruf, der auf I/O-Ereignisse von einem Dateiobjekt (z. B. einem Socket) wartet. 
        #selector.select() gibt eine Liste von (Key, Maske) zurück, wobei jeder key ein Dateiobjekt darstellt und 
        #die Maske ein I/O-Ereignis (z. B. lesen oder schreiben) für dieses Objekt darstellt.
        #Das ist eine Methode, viele verschiedene TCP Verbindungen zu behandeln.
        events = selector.select()  # Auf Ereignisse warten (Lesen oder Schreiben)
        for key, mask in events:  # Alle wartenden Ereignisse durchgehen
            if key.fileobj == server_socket:  # Wenn das Ereignis vom Serversocket kommt
                conn, addr = server_socket.accept()  # Neue Verbindung akzeptieren
                conn.setblocking(False)  # Verbindung nicht blockierend machen
                # Einfache Datenstruktur für die Verbindung erstellen und speichern
                data = types.SimpleNamespace(addr=addr, inb=b"", outb=b"")
                # Ereignisse für Lesen und Schreiben registrieren
                events = selectors.EVENT_READ | selectors.EVENT_WRITE
                # Neue Verbindung beim Selektor registrieren
                selector.register(conn, events, data=data)
                info_Datensatz.delete("1.0", tk.END)
                info_Datensatz.insert(tk.END, f"Connection with client established: {addr}")
                info_type.insert(tk.END, "null")
                info_test.insert(tk.END, "null")

            else:
                conn = key.fileobj
                data = key.data
                try:
                    recv_data = conn.recv(10240)  # Versuch, Daten zu empfangen
                except BlockingIOError:
                    recv_data = b""

                if recv_data:
                    info_tcpdata.delete("1.0", tk.END)
                    info_tcpdata.insert(tk.END, f"Data received from the client: {recv_data.decode()}")#Darstellung der empfangene Daten vom Client
                    data = recv_data.decode().rstrip(',')

                    # Type Nummer und Test Nummer trennen
                    *data_parts, type_no_str, test_no_str = data.split(',')

                    # Daten in ein NumPy-Array umwandeln
                    data_array = np.array(data_parts, dtype=float)

                    # Label-Informationen abrufen, angenommen das Label ist die erste Stelle im Array
                    label = int(data_array[0])

                    # Label entfernen
                    data_without_label = data_array[1:]
                    data_count = len(data_without_label)

                    # Typnummer und Testnummer abrufen
                    last_type_no = type_no_str
                    last_test_no = test_no_str

                    # Zählen der Labels
                    if label == 1:
                        io_count += 1
                    elif label == 0:
                        nio_count += 1

                    # Überprüfen, ob die überschrifte schon geschrieben wurde, wenn nicht, schreiben
                    if not labels_written:
                        with open(TCP_file, 'a', newline='') as csvfile:
                            csv_writer = csv.writer(csvfile)
                            labels = ["Label"] + [str(i) for i in range(1, data_count + 1)] + ["Type"] + ["Test"]
                            csv_writer.writerow(labels)# Überschrifte in die CSV-Datei schreiben
                        labels_written = True

                    # Daten in die CSV-Datei schreiben
                    with open(TCP_file, 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow(data.split(','))  # Daten durch Komma trennen und in CSV schreiben

                    # Textbox aktualisieren
                    received_lines += 1
                    info_Datensatz.delete("1.0", tk.END)
                    info_Datensatz.insert(tk.END, f"Received {received_lines} curves.\n")
                    info_Datensatz.insert(tk.END, f"Feautures: {data_count}\n")
                    info_Datensatz.insert(tk.END, f"IO: {io_count} data\n")
                    info_Datensatz.insert(tk.END, f"NIO: {nio_count} data\n")

                    # Typnummer und Testnummer in separaten Textboxen anzeigen
                    info_type.delete("1.0", tk.END)
                    info_type.insert(tk.END, last_type_no)

                    info_test.delete("1.0", tk.END)
                    info_test.insert(tk.END, last_test_no)

                if recv_data is None:
                    info_Datensatz.delete("1.0", tk.END)
                    info_Datensatz.insert(tk.END, "No data received\n")





def connect(connect_button, disconnect_button, host_entry, port_entry, info_Datensatz,info_type,info_test,info_tcpdata):
    global server_socket, selector, server_running
    # Benutzerdefinierten HOST und PORT abrufen
    HOST = host_entry.get()
    PORT = int(port_entry.get())
    
    # Socket für den Server erstellen und konfigurieren
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen()

    info_Datensatz.delete("1.0", tk.END)

    # Ermittelt die Host und die Portnummer, die der Server tatsächlich überwacht.
    server_address = server_socket.getsockname()
    server_host, server_port = server_address

    info_Datensatz.insert(tk.END,f"Server is listening on ({server_host},{server_port})")
    
    # Selector für das Ereignismanagement erstellen
    selector = selectors.DefaultSelector()
    selector.register(server_socket, selectors.EVENT_READ, data=None)
    server_running = True
    
    # Deaktiviert den Connect button und aktiviert den Disconnect button
    connect_button.config(state=tk.DISABLED)
    disconnect_button.config(state=tk.NORMAL)

    # Startet den Server in einem separaten Thread
    start_server_thread = threading.Thread(target=start_server, args=(server_socket, selector, info_Datensatz,info_type,info_test,info_tcpdata))
    start_server_thread.start()

server_running = True
data_queue = Queue()


def Test_start_server(server_socket, selector, info_Datensatz):
    global conn
    while server_running:
        events = selector.select(timeout=1)  # 1 Sekunde Timeout, um unbegrenztes Warten zu verhindern
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
                    recv_data = conn.recv(10240)  
                except BlockingIOError:
                    pass
                
                if recv_data:
                    print("Received data from client:", recv_data.decode())  
                    received_data = recv_data.decode().rstrip(',')
                    # Eingehende Strings in NumPy-Arrays umwandeln
                    data_array = np.fromstring(received_data, sep=',')
                    data_queue.put(data_array)  # NumPy-Arrays in eine Queue packen
                    data_count = len(data_array)
                    info_Datensatz.delete("1.0", tk.END)
                    info_Datensatz.insert(tk.END, f"Feautures:{data_count}\n")  
                  
                    
                if recv_data == None:
                    info_Datensatz.delete("1.0", tk.END)
                    info_Datensatz.insert(tk.END, "Received no data\n")
    return  conn

def Test_connect(connect_button, disconnect_button, host_entry, port_entry, info_Datensatz):
    global server_socket, selector, server_running,conn

    HOST = host_entry.get()
    PORT = int(port_entry.get())
    
    # Socket für den Server erstellen und konfigurieren
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
    
    # Startet den Server in einem separaten Thread und setzt ihn als Daemon-Thread
    start_server_thread = threading.Thread(target=Test_start_server, args=(server_socket, selector, info_Datensatz))
    start_server_thread.daemon = True  
    conn = start_server_thread.start()

    return conn





def disconnect(connect_button, disconnect_button, info_Datensatz):
    global server_socket, selector, server_running
    
    # Aktiviert den Connect button und deaktiviert den Disconnect button
    connect_button.config(state=tk.NORMAL)
    disconnect_button.config(state=tk.DISABLED)

    server_running = False
    # Schließt den Server-Socket und den Selector
    server_socket.close()
    selector.close()

    info_Datensatz.delete("1.0", tk.END)
    info_Datensatz.insert(tk.END, "Server is stopped\n")




def server_stop(server_socket, selector):
    global server_running
    # Überprüft, ob der Server läuft
    if server_running:
        # Schließt den Server-Socket, falls vorhanden
        if server_socket is not None:
            server_socket.close()
        # Schließt den Selector, falls vorhanden
        if selector is not None:
            selector.close()


def tcp_train_and_visualize(epoch_entry, rate_entry, ax_loss, ax_acc, canvas, text, ge_text,info_text):
    global trained_model
    text.delete(1.0, tk.END)
    ge_text.delete(1.0, tk.END)
    #Die Merkmale sind hier Messungen zu verschiedenen Zeitpunkten
    input_size = train_size 

    # Benutzerdefinierten epochs und Lernrate abrufen
    num_epochs = int(epoch_entry.get())
    learning_rate = float(rate_entry.get())
    hidden_size1 = 4 * input_size
    hidden_size2 = 8 * input_size
    outsize = 1

    info_text.insert(tk.END,"Training model...\n")
    
    #Modell anrufen
    model = NeuralNet(input_size, hidden_size1, hidden_size2, outsize).to(device)
    
    # BCE als Verlustfunktion, Adam als Optimeizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

    # Verlust- und Genauigkeitslisten außerhalb der Schleife definieren
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
   
    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_correct = 0
        running_loss = 0.0
        total_samples = 0

        for i, data in enumerate(train_loader):

            inputs, labels, sample_sizes = data
            inputs = inputs.to(device)
            # Konvertieren der Form des Labels von [batch_size] nach [batch_size, 1].
            labels = labels.unsqueeze(1).to(device)
            labels = labels.float()# Labels in Gleitkommazahlen konvertieren
             
            #Forward Propagation
            outputs = model(inputs)

            #Verlust berechnen
            #nn.BCEWithLogitsLoss() wendet intern automatisch die Sigmoid-Funktion an
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Genauigkeit der Berechnungen und Aufzeichnungen
            predictions = (torch.sigmoid(outputs) >= 0.5).float()
            correct = (predictions == labels).sum().item()
            running_correct += correct
            total_samples += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100.0 * running_correct / total_samples
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # validierung Loop
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
       
        # Aktualisierung der Verlust- und Genauigkeitskurven
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

        canvas.draw_idle()
        canvas.flush_events()

    ge_text.insert(tk.END, f"{val_accuracy}%")        

    info_text.insert(tk.END,"The learning rate can be set according to default values.\nIf the accuracy of the model is not satisfactory, you can try to repeat the training several times or reduce the learning rate and increase the number of training epochs.")
    trained_model = model

    return trained_model






def tcp_save_model( ):
    
    global trained_model 
    
     # Wenn kein trainiertes Modell vorhanden ist, passiert nichts.
    if trained_model is None:
        print("No trained model available.")
        return
    
    # Aufforderung an den Benutzer, einen Speicherort auszuwählen
    file_path = filedialog.asksaveasfilename(defaultextension=".pth", filetypes=[("Model files", "*.pth")])
    
    # Wenn kein Pfad vorhanden ist, passiert nichts.
    if not file_path:
        return
    
    # nur Modell Parameter speichern
    torch.save(trained_model.state_dict(), file_path)
    print("Model saved successfully.")
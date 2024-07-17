import tkinter as tk
from tkinter import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from tkinter import filedialog


# Gerätekonfigurationen für das KI-Training
#cuda für GPU
#CPU für CPU
#device = torch.device('cpu')
device = torch.device('cuda')

# Global Variante
trained_model = None
num_classes = 2 #Binäre Klassifikation


class KurvenDataset(Dataset):

    def __init__(self,csv_file_path):
        # Daten initialisieren, herunterladen, etc.
        # Lesen mit Numpy oder Pandas
        xy = np.loadtxt(csv_file_path, delimiter=',', dtype=np.float32, skiprows=1) 
        self.n_samples = xy.shape[0]

        # hier ist die erste Spalte die Labels, der Rest sind die Merkmale
        self.x_data = torch.from_numpy(xy[:, 1:]) # size [n_samples, n_features]
        self.y_data = torch.from_numpy(xy[:, 0]).to(torch.long)
              
        # Berechnung der Probengrößen
        self.sample_sizes = [len(sample) for sample in self.x_data]

     # Unterstützung des index, so dass dataset[i] verwendet werden kann, um die i-te Probe zu erhalten
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.sample_sizes[index]

    # len(dataset) kann aufgerufen werden, um die Probengröße zurückzugeben
    def __len__(self):
        return self.n_samples
    
    # Funktion zur Ermittlung der Anzahl der einzelnen Klassen im Datensatz
    def class_counts(self):
        class_count = {}
        for label in self.y_data.tolist():
            if label not in class_count:
                class_count[label] = 1
            else:
                class_count[label] += 1
        return class_count

class NeuralNet(nn.Module):
    #Entwerfen der Struktur des neuronalen Netzes und Erhöhung der Anzahl der Neuronen in Hidden Layer, 
    #wenn es sich um ein komplexeres Klassifizierungsproblem handelt
    def __init__(self, input_size,hidden_size1,hidden_size2,outsize):
        super(NeuralNet, self).__init__()
        hidden_size1 = 4 * input_size
        hidden_size2 = 8 * input_size
        self.input_size = input_size
        outsize = 1
        self.l1 = nn.Linear(input_size, hidden_size1) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size1,outsize)
        # self.relu2 = nn.LeakyReLU()
        # self.l3 = nn.Linear(hidden_size2, outsize)
    
    #Forward Propagation
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # out = self.relu2(out)
        # out = self.l3(out)
        return out

def on_choose_traindata_button_click(batch_entry, csv_button, Trainsinfo_text):
    global train_size,train_loader
    # Benutzerdefinierten CSV-Dateipfad abrufen
    csv_file_path = filedialog.askopenfilename(title="Select Traindata", filetypes=[("CSV Files", "*.csv")])
    batch_size = int(batch_entry.get())
    
    if csv_file_path:
        # Der Dateipfad wird auf der Benutzeroberfläche angezeigt, und der Button wird nach Auswahl der Datei grün.
        csv_button.configure(bg="green")
        Trainsinfo_text.delete(1.0, tk.END) 
        
        dataset1 = KurvenDataset(csv_file_path)
        train_loader = torch.utils.data.DataLoader(dataset=dataset1,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
        
        _,_,train_size = KurvenDataset.__getitem__(dataset1,1)
        #Hier ist KurvenDataset eine Klasse, also müssen wir eine Instanz als ersten Parameter übergeben.
     
        Trainsinfo_text.insert(tk.END,f"{train_size} Feautures\n")
        class_counts = dataset1.class_counts()
        
        # Definiere ein Wörterbuch, um Bezeichnungen den entsprechenden Namen zuzuordnen.
        label_mapping = {1: "IO", 0: "NIO"}

        # Ausgabe gemappter Namen
        for label, count in class_counts.items():
            if label in label_mapping:
                label_name = label_mapping[label]
            else:
                label_name = str(label)  # Wenn die Bezeichnung nicht in der Zuordnung enthalten ist, wird die ursprüngliche Bezeichnung verwendet.
            Trainsinfo_text.insert(tk.END, f"{label_name} Data: {count} samples\n")

    return train_loader, train_size


def on_choose_testdata_button_click(batch_entry,text,csv_button):
   
    global validation_loader
    batch_size = int(batch_entry.get())
    text.delete(1.0, tk.END)
    # Benutzerdefinierten CSV-Dateipfad abrufen
    csv_file_path = filedialog.askopenfilename(title="Select Testdata", filetypes=[("CSV Files", "*.csv")])

    if csv_file_path:
        # Der Dateipfad wird auf der Benutzeroberfläche angezeigt, und der Button wird nach Auswahl der Datei grün.
        csv_button.configure(bg="green")
        dataset = KurvenDataset(csv_file_path)
        validation_loader = torch.utils.data.DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
        
        _,_,sample_size = KurvenDataset.__getitem__(dataset,1)
       
        text.insert(tk.END,f"{sample_size} Merkmale\n")

        class_counts = dataset.class_counts()
        label_mapping = {1: "IO", 0: "NIO"}

        for label, count in class_counts.items():
            if label in label_mapping:
                label_name = label_mapping[label]
            else:
                label_name = str(label)  
            text.insert(tk.END, f"{label_name} Daten: {count} Proben\n")
    
    return validation_loader,sample_size,label,count


# Erstellen einer Funktion, die den Trainings- und Visualisierungsprozess kapselt
def train_and_visualize( epoch_entry, rate_entry, ax_loss, ax_acc, canvas, text, ge_text):
    global trained_model
    text.delete(1.0, tk.END)
    ge_text.delete(1.0, tk.END)
    # Benutzerdefinierten epochs und Lernrate abrufen
    input_size = train_size
    num_epochs = int(epoch_entry.get())
    learning_rate = float(rate_entry.get())
    
    text.insert(tk.END,"Training model...\n")

    hidden_size1 = 4 * input_size
    hidden_size2 = 8 * input_size
    outsize = 1
    #Modell anrufen
    model = NeuralNet(input_size,hidden_size1,hidden_size2,outsize).to(device)

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
            inputs, labels, sample_size = data
            inputs = inputs.to(device)
            # Konvertieren der Form des Labels von [batch_size] nach [batch_size, 1].
            # BCELoss erwartet, dass die Eingaben (Vorhersagen) und die Labels die gleiche Form haben.
            # Wenn die Vorhersagen die Form [batch_size, 1] haben, müssen auch die Labels diese Form haben.
            #z.B [1, 0, 0] shape: [3] ---> [[1], [0], [0]] shape: [3,1]          
            labels = labels.unsqueeze(1).to(device)  # Form des Labels ändern und auf das Gerät verschieben
            labels = labels.float()  # Labels in Gleitkommazahlen konvertieren

            #Forward Propagation
            outputs = model(inputs)
            #Verlust berechnen
            #nn.BCEWithLogitsLoss() wendet intern automatisch die Sigmoid-Funktion an
            loss = criterion(outputs, labels)
            #Adam
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Aufgezeichnete Verlustwerte
            running_loss += loss.item()

            # Genauigkeit der Berechnungen und Aufzeichnungen
            predictions = (torch.sigmoid(outputs) >= 0.5).float()  # Hier werden benutzerdefinierte Grenzewerte verwendet
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
                inputs, labels, sample_size = data
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

        # Akutualisierung des Zeichens
        canvas.draw_idle()
        canvas.flush_events()

    ge_text.insert(tk.END, f"{val_accuracy}%")        

    text.insert(tk.END,"The learning rate can be set according to default values.\nIf the accuracy of the model is not satisfactory, you can try to repeat the training several times or reduce the learning rate and increase the number of training epochs.")

    trained_model = model

    return trained_model


def save_model( ):
    
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



def load_ai_model(kern_button,KI_text):
    ai_model_path = filedialog.askopenfilename(title="Select AI Model", filetypes=[("Model files", "*.pth")])
       
    if ai_model_path:
        # Der Dateipfad wird auf der Benutzeroberfläche angezeigt, und der Button wird nach Auswahl der Datei grün.
        kern_button.configure(bg="green")
        KI_text.delete(1.0, END)  
        KI_text.insert(END, ai_model_path)
    return ai_model_path





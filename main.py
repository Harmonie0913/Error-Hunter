import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from mode1 import Mode1UIManager
from mode2 import Mode2UIManager

class ErrorHunterApp:
        
    def __init__(self, window):
        #Initialisierung der Fenstereigenschaften
        self.window = window
        self.current_ui_manager = self
        self.frame_top = None
        self.window.title("Error Hunter")
        self.window.iconbitmap(r"C:\Users\yangx\OneDrive - KUKA AG\EH\KI.ico")   
        self.create_widgets()

    def create_widgets(self):
        # Dropdown-Menüs erstellen
        menubar = tk.Menu(self.window)
        mode_menu = tk.Menu(menubar, tearoff=0)
        mode_menu.add_command(label="New Training", command=self.create_mode1_window)
        mode_menu.add_command(label="New Test", command=self.create_mode2_window)
        menubar.add_cascade(label="Select Mode", menu=mode_menu)
        self.window.config(menu=menubar)

        # KUKA Logo in der oberen linken Ecke einfügen
        new_size = (140, 24)
        self.insert_image(new_size)

        # Titel "Error detection" in der oberen rechten Ecke einfügen
        text = "Error detection"
        font = ('times', 15, 'bold')
        font1=("Helvetica", 14)
        font2=("Helvetica", 16)
        color = "red"
        color1 = 'black'
        self.frame_top = tk.Frame(self.window)
        self.frame_top.pack(side="right", anchor="ne")
        self.ProgName = tk.Label(self.frame_top, text=text, font=font, fg=color)
        
        self.ProgName.pack(side="right", anchor="ne")
         
        # Englische Anleitungen in der Mitte einfügen 
        self.frame1 = tk.Frame(self.window)
        self.frame1.pack(side="top", pady=35)        
        self.ProgName1 = tk.Label(self.frame1, text="Instructions", font=font2, fg=color1)
        self.ProgName2 = tk.Label(self.window, text="Training mode\nTraining new models \nfor different types of curves ", font=font1, fg=color1)
        self.sepa = tk.Label(self.window)
        self.ProgName3 = tk.Label(self.window, text="Test mode\nExecution of error detection \nby AI model", font=font1, fg=color1)

        self.ProgName1.pack(side="top")
        self.ProgName2.pack(side="top")
        self.sepa.pack(side="top",pady=10)
        self.ProgName3.pack(side="top")

        self.Exit_button = tk.Button(self.window, text="Quit", command=self.window.quit)
        self.Exit_button.pack(side="bottom")

    def insert_image(self,new_size):
        
        image_path = r"C:\Users\yangx\OneDrive - KUKA AG\EH\KUKA_Logo.png"  # Kann durch den gewünschten Bildpfad ersetzt werden

        # Öffnung des Bilds und in ein PhotoImage-Objekt konvertiert wird
        original_image = Image.open(image_path)
        resized_image = original_image.resize(new_size)  # Größenänderung von Bildern
        photo = ImageTk.PhotoImage(resized_image)
        self.image_label = tk.Label(self.frame_top)
        # Bilder in Label einfügen
        self.image_label.config(image=photo)
        self.image_label.image = photo 
        # Das Bild in der oberen linken Ecke einfügen 
        self.image_label.pack(side="left", anchor="nw")
    
    #Zum Umschalten zwischen zwei Modi
    def create_mode1_window(self):
        mode1_ui = Mode1UIManager(self.window)
        #Ursprünglich erzeugte Fenster in Training Mode: Training Funktion
        mode1_ui.create_Training_widgets()

    def create_mode2_window(self):
        mode2_ui = Mode2UIManager(self.window)
        #Ursprünglich erzeugte Fenster in Test Mode: Automatisch Funktion
        mode2_ui.create_Automatisch_widgets()

if __name__ == "__main__":  # Hauptprogrammstart, um sicherzustellen, dass der Code nur ausgeführt wird, wenn das Skript direkt ausgeführt wird
    window = tk.Tk()  # Erstellen eines Hauptfensters für die GUI
    app = ErrorHunterApp(window)  # Initialisieren der ErrorHunterApp mit dem Hauptfenster
    window.mainloop()  # Starten der Hauptschleife des Fensters, damit die GUI auf Benutzerinteraktionen reagiert

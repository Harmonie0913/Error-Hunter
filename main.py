import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from mode1 import Mode1UIManager
from mode2 import Mode2UIManager

class ErrorHunterApp:
        
    def __init__(self, window):
        self.window = window
        self.current_ui_manager = self
        self.frame_top = None
        #self.window.geometry("400x300")
        self.window.title("Error Hunter")
        self.window.iconbitmap(r"C:\Users\yangx\OneDrive - KUKA AG\EH\KI.ico")
        self.window.geometry("600x360")       
        self.create_widgets()

    def create_widgets(self):
        # 创建菜单
        menubar = tk.Menu(self.window)
        mode_menu = tk.Menu(menubar, tearoff=0)
        mode_menu.add_command(label="New Train", command=self.create_mode1_window)
        mode_menu.add_command(label="New Test", command=self.create_mode2_window)
        menubar.add_cascade(label="Select Mode", menu=mode_menu)
        self.window.config(menu=menubar)

        # 插入图像
        new_size = (140, 24)
        self.insert_image(new_size)

        # 插入 ProgName 的 Label
        text = "Fehlererkennung"
        font = ('times', 15, 'bold')
        font1=("Helvetica", 14)
        font2=("Helvetica", 16)
        color = "red"
        color1 = 'black'
        self.frame_top = tk.Frame(self.window)
        self.frame_top.pack(side="top", fill="x", pady=0)
        self.ProgName = tk.Label(self.frame_top, text=text, font=font, fg=color)
        self.frame1 = tk.Frame(self.window)
        self.frame1.pack(side="top", fill="x", pady=10)        
        self.ProgName1 = tk.Label(self.frame1, text="Anleitung", font=font2, fg=color1)
        self.ProgName2 = tk.Label(self.window, text="Train mode\nTraining neuer Modelle  \nfür verschiedene Arten von Kurven ", font=font1, fg=color1)
        self.sepa = tk.Label(self.window)
        # self.separator_frame = tk.Frame(self.frame_top)
        # self.separator_frame.pack(side="top",pady=30)
        self.ProgName3 = tk.Label(self.window, text="Test mode\nDurchführung der Fehlererkennung \ndurch KI Modell", font=font1, fg=color1)
        
        # 显示 ProgName 的 Label
        self.ProgName.pack(side="right")
        self.ProgName1.pack(side="top", anchor="w")
        self.ProgName2.pack(side="top", anchor="w")
        self.sepa.pack(side="top",pady=10)
        self.ProgName3.pack(side="top", anchor="w")

        self.Exit_button = tk.Button(self.window, text="Quit", command=self.window.quit)
        self.Exit_button.pack(side="bottom")

    def insert_image(self,new_size):
        image_path = r"C:\Users\yangx\OneDrive - KUKA AG\EH\KUKA_Logo.png"  # 替换为你的图片路径

        # 打开图像并将其转换为 PhotoImage 对象
        original_image = Image.open(image_path)
        resized_image = original_image.resize(new_size)  # 调整图像大小
        photo = ImageTk.PhotoImage(resized_image)
        self.image_label = tk.Label(self.frame_top)
        # 在标签中插入图片
        self.image_label.config(image=photo)
        self.image_label.image = photo  # 保持对 PhotoImage 的引用
        self.image_label.pack(side="left", anchor="ne")
    
    def destroy_widgets(self):
        self.image_label.pack_forget()
        self.ProgName.pack_forget()
        self.ProgName1.pack_forget()
        self.ProgName2.pack_forget()
        self.ProgName3.pack_forget()
        self.frame_top.pack_forget()
        self.frame1.pack_forget()
        self.sepa.pack_forget()
    


    def create_mode1_window(self):
        mode1_ui = Mode1UIManager(self.window)
        mode1_ui.create_Training_widgets()

    def create_mode2_window(self):
        mode2_ui = Mode2UIManager(self.window)
        mode2_ui.create_Automatisch_widgets()

if __name__ == "__main__":
    window = tk.Tk()
    app = ErrorHunterApp(window)
    window.mainloop()
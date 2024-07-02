import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import numpy as np
import json

class DataPreprocessor:
    def __init__(self):
        self.preprocessing_steps = []

    def add_step(self, operation, indices):
        self.preprocessing_steps.append({'operation': operation, 'indices': indices})

    def remove_step(self, index):
        if 0 <= index < len(self.preprocessing_steps):
            self.preprocessing_steps.pop(index)

    def move_step_up(self, index):
        if 1 <= index < len(self.preprocessing_steps):
            self.preprocessing_steps[index], self.preprocessing_steps[index-1] = \
                self.preprocessing_steps[index-1], self.preprocessing_steps[index]

    def move_step_down(self, index):
        if 0 <= index < len(self.preprocessing_steps) - 1:
            self.preprocessing_steps[index], self.preprocessing_steps[index+1] = \
                self.preprocessing_steps[index+1], self.preprocessing_steps[index]

    def process(self, data):
        processed_data = []
        for step in self.preprocessing_steps:
            operation = step['operation']
            indices = step['indices']
            if operation == 'mean':
                processed_data.append(np.mean(data[indices]))
            elif operation == 'median':
                processed_data.append(np.median(data[indices]))
            elif operation == 'std':
                processed_data.append(np.std(data[indices]))
            elif operation == 'ratio':
                processed_data.append(data[indices[0]] / data[indices[1]])
            else:
                processed_data.append(data[indices[0]])
        return np.array(processed_data)

    def save(self, filename):
        with open(filename, 'w') as file:
            json.dump(self.preprocessing_steps, file)

    def load(self, filename):
        with open(filename, 'r') as file:
            self.preprocessing_steps = json.load(file)
        return self


class DataPreprocessorWindow(tk.Toplevel):
    def __init__(self, window):
        self.window = tk.Toplevel(window)
        #self.window = window
        self.preprocessor = None
        self.window.iconbitmap(r"C:\Users\yangx\OneDrive - KUKA AG\EH\KI.ico")
        #self.window.geometry("600x685")

        #self.create_widgets()


    def create_widgets(self):
       
        self.window.title('Data Preprocessor Designer')
        self.preprocessor = DataPreprocessor()

        self.step_listbox = tk.Listbox(self.window)
        self.step_listbox.pack(fill=tk.BOTH, expand=True)

        self.button_frame = ttk.Frame(self.window)
        self.button_frame.pack(fill=tk.X)

        self.add_mean_button = ttk.Button(self.button_frame, text='Add Mean', command=lambda: self.add_step('mean'))
        self.add_mean_button.pack(side=tk.LEFT)

        self.add_median_button = ttk.Button(self.button_frame, text='Add Median', command=lambda: self.add_step('median'))
        self.add_median_button.pack(side=tk.LEFT)

        self.add_std_button = ttk.Button(self.button_frame, text='Add Std Dev', command=lambda: self.add_step('std'))
        self.add_std_button.pack(side=tk.LEFT)

        self.add_ratio_button = ttk.Button(self.button_frame, text='Add Ratio', command=lambda: self.add_step('ratio'))
        self.add_ratio_button.pack(side=tk.LEFT)

        self.add_value_button = ttk.Button(self.button_frame, text='Add Value', command=lambda: self.add_step('value'))
        self.add_value_button.pack(side=tk.LEFT)

        self.save_button = ttk.Button(self.button_frame, text='Save Steps', command=self.save_steps)
        self.save_button.pack(side=tk.LEFT)

        self.load_button = ttk.Button(self.button_frame, text='Load Steps', command=self.load_steps)
        self.load_button.pack(side=tk.LEFT)

        self.remove_button = ttk.Button(self.button_frame, text='Remove Step', command=self.remove_step)
        self.remove_button.pack(side=tk.LEFT)

        self.up_button = ttk.Button(self.button_frame, text='Move Up', command=self.move_up)
        self.up_button.pack(side=tk.LEFT)

        self.down_button = ttk.Button(self.button_frame, text='Move Down', command=self.move_down)
        self.down_button.pack(side=tk.LEFT)

    def add_step(self, step_type):
        indices = simpledialog.askstring('Indices', 'Enter indices (comma-separated):')
        if indices:
            indices = list(map(int, indices.split(',')))
            self.preprocessor.add_step(step_type, indices)
            self.update_step_listbox()

    def update_step_listbox(self):
        self.step_listbox.delete(0, tk.END)
        for step in self.preprocessor.preprocessing_steps:
            self.step_listbox.insert(tk.END, f"{step['operation']} {step['indices']}")

    def save_steps(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            self.preprocessor.save(file_path)
            messagebox.showinfo('Info', 'Preprocessing steps saved successfully')

    def load_steps(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            self.preprocessor = DataPreprocessor().load(file_path)
            DataPreprocessorWindow.current_preprocessor = self.preprocessor
            self.update_step_listbox()
            messagebox.showinfo('Info', 'Preprocessing steps loaded successfully')

    def remove_step(self):
        selected_index = self.step_listbox.curselection()
        if selected_index:
            self.preprocessor.remove_step(selected_index[0])
            self.update_step_listbox()

    def move_up(self):
        selected_index = self.step_listbox.curselection()
        if selected_index:
            self.preprocessor.move_step_up(selected_index[0])
            self.update_step_listbox()

    def move_down(self):
        selected_index = self.step_listbox.curselection()
        if selected_index:
            self.preprocessor.move_step_down(selected_index[0])
            self.update_step_listbox()



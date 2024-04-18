import tkinter as tk

def create_new_page():
    new_page = tk.Toplevel(root)
    new_page.title("New Page")
    label = tk.Label(new_page, text="This is a new page!")
    label.pack()

root = tk.Tk()
root.title("Main Window")

# 创建菜单栏
menubar = tk.Menu(root)

# 创建菜单项
file_menu = tk.Menu(menubar, tearoff=0)
file_menu.add_command(label="New Page", command=create_new_page)

# 将菜单项添加到菜单栏
menubar.add_cascade(label="File", menu=file_menu)

# 将菜单栏添加到主窗口
root.config(menu=menubar)

root.mainloop()

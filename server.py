import socket
import threading
import tkinter as tk
import selectors
import types

def get_connect():
    Host = HOST_Entry.get()
    Port = int(POST_Entry.get())
    return Host, Port

def start_server():
    global server_socket, selector
    HOST, PORT = get_connect()
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen()

    text_box.insert(tk.END, "Server is running. Waiting for connections...\n")

    selector = selectors.DefaultSelector()
    selector.register(server_socket, selectors.EVENT_READ, data=None)

    while True:
        events = selector.select()
        for key, mask in events:
            if key.fileobj == server_socket:
                conn, addr = server_socket.accept()
                text_box.insert(tk.END, "Connected by: " + str(addr) + "\n")
                conn.setblocking(False)
                data = types.SimpleNamespace(addr=addr, inb=b"", outb=b"")
                events = selectors.EVENT_READ | selectors.EVENT_WRITE
                selector.register(conn, events, data=data)
            else:
                handle_client_data(key, mask, selector)

def handle_client_data(key, mask, selector):
    sock = key.fileobj
    data = key.data
    if mask & selectors.EVENT_READ:
        recv_data = sock.recv(1024)
        # 根据数据的特定格式或标志来确定应该对数据进行何种操作
        if recv_data.startswith(b'TRAIN:'):
            # 提取需要训练的随机数数据
            if recv_data:
                received_messages = recv_data.decode().split(',')  # 假设数据以逗号分隔
                text_box.insert(tk.END, "Received: \n")
                for message in received_messages:
                    text_box.insert(tk.END, message + "\n")
            else:
                selector.unregister(sock)
                sock.close()

            if mask & selectors.EVENT_WRITE:
                if data.outb:
                    sent = sock.send(data.outb)
                    data.outb = data.outb[sent:]

        elif recv_data.startswith(b'TEST:'):
            if recv_data:
                received_messages = recv_data.decode().split(',')  # 假设数据以逗号分隔
                responses = []
                for message in received_messages:
                    try:
                        prefix, num_str = message.split(':')  # 分割前缀和数字
                        num = int(num_str)
                        if num > 500:
                            responses.append("IO")
                        else:
                            responses.append("NIO")
                    except ValueError:
                        print("Invalid message:", message)
                response_data = ','.join(responses).encode()  # 将响应数据转换为字节形式
                sock.sendall(response_data)  # 发送响应数据给客户端
                text_box.insert(tk.END, "Received: \n")
                for message in received_messages:
                    text_box.insert(tk.END, message + "\n")
            else:
                selector.unregister(sock)
                sock.close()

            if mask & selectors.EVENT_WRITE:
                if data.outb:
                    sent = sock.send(data.outb)
                    data.outb = data.outb[sent:]
        else:
            return "Invalid command"

def clean_textbox():
    text_box.delete("1.0", tk.END)  # 清空文本框

def connect():
    global connect_button, disconnect_button
    connect_button.config(state=tk.DISABLED)
    disconnect_button.config(state=tk.NORMAL)
    start_server_thread = threading.Thread(target=start_server)
    start_server_thread.start()

def disconnect():
    global server_socket, selector, connect_button, disconnect_button
    connect_button.config(state=tk.NORMAL)
    disconnect_button.config(state=tk.DISABLED)
    server_socket.close()
    selector.close()
    text_box.insert(tk.END, "Server Dissconnect\n")

server_window = tk.Tk()
server_window.title("Server")

HOST_Label = tk.Label(server_window,text='SERVER HOST:')
HOST_Label.pack()

HOST_Entry = tk.Entry(server_window,width=15)
HOST_Entry.pack()

POST_Label = tk.Label(server_window,text='SERVER PORT:')
POST_Label.pack()

POST_Entry = tk.Entry(server_window,width=15)
POST_Entry.pack()

connect_button = tk.Button(server_window, text="Connect", command=connect)
connect_button.pack()

disconnect_button = tk.Button(server_window, text="Disconnect", command=disconnect, state=tk.DISABLED)
disconnect_button.pack()

text_box = tk.Text(server_window)
text_box.pack()

quit_button = tk.Button(server_window, text="Quit", command=server_window.quit)
quit_button.pack()

clean_button = tk.Button(server_window, text="clean", command=clean_textbox)
clean_button.pack()

server_window.mainloop()

from tkinter import *
import os
from turtle import width

def get_file_list():
    filepath = entry_input_path.get()
    filelist = []
    # 遍历filepath下所有文件，包括子目录
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath, fi)
        if not (os.path.isdir(fi_d)):
            filelist.append(fi)
    list_files.delete(0,END)
    for item in files:
        list_files.insert(0,item)


root = Tk()
root.geometry('450x300')

## IO
frame_paths = Frame(root)
lable_input = Label(frame_paths,text="输入目录")
entry_input_path = Entry(frame_paths,width=40)
entry_input_path.insert(0,r".\dataset\normal")
button_input_select = Button(frame_paths,text="...")

lable_input.grid(row=0,column=0)
entry_input_path.grid(row=0,column=1)
button_input_select.grid(row=0,column=2)

lable_output = Label(frame_paths,text="输入目录")
entry_output_path = Entry(frame_paths,width=40)
entry_output_path.insert(0,r".\dataset\CWRU\dataset.npy")
button_output_select = Button(frame_paths,text="...")

lable_output.grid(row=1,column=0)
entry_output_path.grid(row=1,column=1)
button_output_select.grid(row=1,column=2)

frame_paths.pack()

## read dir
frame_files = Frame(root)
list_files = Listbox(frame_files,width=40)
list_files.pack()

button_load = Button(frame_paths,text="加载目录",command=get_file_list)
button_load.grid(row=2,column=1)

frame_files.pack()

button_load_certain_file = Button(frame_files,text="加载文件")
button_load_certain_file.pack()

## read files



root.mainloop()                 # 进入消息循环

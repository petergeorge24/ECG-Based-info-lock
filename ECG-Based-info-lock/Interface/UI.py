import tkinter as tk
from tkinter import *
from tkinter import filedialog, ttk

import cv2
from PIL import ImageTk, ImageSequence,Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from HCISingleSignals import load_signal
from Run import run , similarity

root = Tk()
root.title("ECG based Info_Lock")
root.configure(background='gray')
root.resizable(False, False)
root.geometry("1000x600")

root.filename = filedialog.askopenfilename(title="Select a file")

# Read Signal
signal = load_signal(root.filename[0:-4])

# plot Signal on GUI
fig = Figure(figsize=(6, 4))
ax = fig.add_subplot(111)
ax.plot(signal)
ax.set_xlim(0, 3000)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()
canvas.get_tk_widget().configure(width=1000, height=300)
canvas.draw()

start = 0
end = 3000
iterations = 0


def update_plot():
    global iterations
    global start, end
    ax.clear()
    ax.plot(signal)
    ax.set_xlim(start + 100, end + 100)

    start += 100
    end += 100

    canvas.draw()
    iterations += 1

    # Schedule the next plot update after a specified interval (e.g., 1000 milliseconds)
    if iterations < 6000:
        root.after(20, update_plot)


update_plot()

# People on the system
Users = {0: "سباجتي", 1: "بطة", 2: "فلفل", 3: "صلاح"}
Names = {0: "سباجتي", 1: "بطة", 2: "فلفل", 3: "صلوحة"}
Ages = {0: "a soul of a baby", 1: "22 but looks old and grumpy", 2: "CIA privileged info.", 3: "الايفيهات خلصت"}


# Create another container for everything but signal plot
container2 = tk.Frame(root, bg="white")
container2.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

img_lb = tk.Label(root)
img_lb.pack(pady=0)

def setimg(path):
    global img
    img = tk.PhotoImage(file=path)
    img_lb.config(image=img)


# Login button
def execute():
    person_index = run(signal)
    sim = similarity(signal)
    if sim<92:
        Label(container2, text='مش فاكرك يااض', background='white', font=("Helvetica", 16)).grid(row=3,column=2,pady=0)
        setimg("images.png")

    else:
        message = 'الله!! ابو {name} ابن الناظر'.format(name=Users[person_index])
        Label(container2, text=message, background='white', font=("Helvetica", 16)).grid(row=3,column=2, pady=0)

        message2 = 'Name: {name}'.format(name=Names[person_index])
        Label(container2, text=message2, background='white', font=("Helvetica", 16)).grid(row=5, column=2, pady=0)

        message3 = 'Age: {age}'.format(age=Ages[person_index])
        Label(container2, text=message3, background='white', font=("Helvetica", 16)).grid(row=6, column=2, pady=0)

        setimg("f.png")


Button(container2, text='افتح يا سمسم', width=10, font=("Helvetica", 16), command=execute).grid(row=2, column=2, padx=435)


mainloop()

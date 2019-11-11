import PIL
from PIL import Image,ImageTk
import pytesseract
import cv2
from tkinter import *


width, height = 800, 600
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

root = Tk()
root.bind('<Escape>', lambda e: root.quit())
lmain = Label(root)
lmain.pack()

def show_frame():
	_, frame = cap.read()
	frame = cv2.flip(frame, 1)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	ret, cv2image = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	
	frame [:,:,0] = cv2image/255 * frame [:,:,0] 
	frame [:,:,1] = cv2image/255 * frame [:,:,1]
	frame [:,:,2] = cv2image/255 * frame [:,:,2]


	img = PIL.Image.fromarray(frame)
	imgtk = ImageTk.PhotoImage(image=img)
	lmain.imgtk = imgtk
	lmain.configure(image=imgtk)
	lmain.after(10, show_frame)

show_frame()
root.mainloop()


import warnings
from pathlib import Path
import cv2, os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import preprocessing
from keras.models import model_from_json
from keras.models import load_model

import pydensecrf.densecrf as dcrf

from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, softmax_to_unary, unary_from_softmax


# Interface
from tkinter import *        
from PIL import ImageTk, Image
import easygui
from tkinter import messagebox
from tkinter import filedialog


class App:
	def __init__(self, root):
		self.root=root
		self.error = 0

		self.segmentation = Text(self.root, height=30,bd=1, relief="sunken", width=80)
		self.segmentation.tag_configure("center", justify='center')
		self.segmentation.insert("1.0", "\n\n\n\n\n\n\n\n\n\n\n\n\n\nSegmentation Image")
		self.segmentation.tag_add("center", "1.0", "end")
		self.segmentation.pack(side=LEFT)
		self.img_seg = Label()
		self.img_seg.pack()

		self.topFrame = Frame(self.root,bd=1, relief="sunken")
		self.topFrame.pack()
		self.origin = Text(self.topFrame, height=12, width=50)
		self.origin.tag_configure("center", justify='center')
		self.origin.insert("1.0", "\n\n\n\n\n\nImage")
		self.origin.tag_add("center", "1.0", "end")
		self.origin.pack(side=LEFT,padx=5)
		self.img_origin = Label()
		self.img_origin.pack()




		self.centerFrame = Frame(self.root,height=20,width=50)
		self.centerFrame.pack()
		
		self.firstcenter = Frame(self.centerFrame)
		self.firstcenter.pack(side=LEFT)

		S1 = Label(self.firstcenter, text="Scale:(0.7)")
		S1.pack(side = LEFT,padx=5)
		self.scale_entry = Entry(self.firstcenter)
		self.scale_entry.config(width=7)
		self.scale_entry.pack(side = RIGHT, padx=10)



		self.CRF_but = Button(self.centerFrame, text="CRFs",state=DISABLED, command = self.get_CRFs)
		self.CRF_but.pack(side=RIGHT, padx=10)

		self.secondcenter = Frame(self.centerFrame)
		self.secondcenter.pack(side=RIGHT)

		I1 = Label(self.secondcenter, text="Inference:(20)")
		I1.pack(side = LEFT)
		self.inference_entry = Entry(self.secondcenter)
		self.inference_entry.config(width=7)
		self.inference_entry.pack(side = RIGHT)


	
		


		



		#img = PhotoImage(file = "D:/LV/FCN/FCN6/img.png")

		self.bottomFrame = Frame(self.root)
		self.bottomFrame.pack(side = BOTTOM)
		self.camera = Button(self.bottomFrame, text="Open Camera",state=NORMAL, command = self.get_Camera)
		self.camera.pack(side=LEFT, pady=25,padx=5)

		self.stop = Button(self.bottomFrame, text="Stop Camera",state=DISABLED,command = self.stopAnimation)
		self.stop.pack(side=LEFT, pady=25,padx=5)

		self.im_vi = Button(self.bottomFrame, text="Image",state=NORMAL,command = self.get_Image_Video)
		self.im_vi.pack(side=LEFT, pady=25,padx=5)

		self.save = Button(self.bottomFrame, text="Save",state=DISABLED, command = self.get_Save)
		self.save.pack(side=LEFT, pady=25,padx=5)





		self.bagrFrame = Frame(self.root,bd=1, relief="sunken")
		self.bagrFrame.pack(side=BOTTOM)


		self.bagrFrame.grid_rowconfigure(0, weight=1)
		self.bagrFrame.grid_columnconfigure(0, weight=1)

		self.scroll = Scrollbar(self.bagrFrame, orient=VERTICAL)
		self.text = Text(self.bagrFrame, wrap=NONE, bd=0,
		           height=7,width=40)

		self.text.grid(row=0, column=0, sticky=N+S+W)
		self.Map_but = Button(self.bagrFrame, text="Map",state=DISABLED, command = self.Map)
		self.Map_but.grid(row=0, column=1,sticky=E)


		self.L = Listbox(self.text, width = 50, height = 5)
		self.gifsdict = {}

		path = "./background/"
		background = os.listdir(path)
		for i in background:
			photo = ImageTk.PhotoImage(Image.open(path+i).resize((100, 100), Image.ANTIALIAS))
			self.gifsdict[i] = photo
			self.L.insert(END, i)

		self.L.pack()
		self.img = Label()
		self.img.pack(side=BOTTOM, pady=5)
		self.L.bind('<ButtonRelease-1>', self.list_entry_clicked)
		

	def give_color_to_seg_img(self,seg,n_classes):
	    '''
	    seg : (input_width,input_height,3)
	    '''
	    
	    if len(seg.shape)==3:
	        seg = seg[:,:,0]
	    seg_img = np.zeros( (seg.shape[0],seg.shape[1],3) ).astype('float')
	    colors = sns.color_palette("hls", n_classes)
	    
	    for c in range(n_classes):
	        segc = (seg == c)
	        seg_img[:,:,0] += (segc*( colors[c][0] ))
	        seg_img[:,:,1] += (segc*( colors[c][1] ))
	        seg_img[:,:,2] += (segc*( colors[c][2] ))

	    seg_img =(seg_img * 255).astype(np.uint8)
	    return(seg_img)






	def list_entry_clicked(self,*ignore):
		

		self.CRF_but.configure(state=DISABLED)
		

		self.imgname = self.L.get(self.L.curselection()[0])
		self.img.config(image=self.gifsdict[self.imgname])
		

		


	def stopAnimation(self):
		self.cap.release()
		self.camera.configure(state=NORMAL)
		self.im_vi.configure(state=NORMAL)
		self.save.configure(state=NORMAL)
		self.stop.configure(state=DISABLED)
		self.CRF_but.configure(state=NORMAL)
		self.Map_but.configure(state=NORMAL)

		self.data_upload_image_seg = cv2.cvtColor(self.data_upload_image_seg, cv2.COLOR_BGR2RGB)
		self.data_upload_image_seg = cv2.resize(self.data_upload_image_seg, ( 224 , 224 ))

		self.data_upload_image_seg = np.expand_dims(self.data_upload_image_seg,axis=0 )
		self.y_pred = model.predict(self.data_upload_image_seg)
		self.data_original = self.data_upload_image_seg
		self.y_predi = np.squeeze(np.argmax(self.y_pred, axis=3), axis=0)
		self.data_upload_image_seg = self.give_color_to_seg_img(self.y_predi,2)
		self.saved_upload_image_seg = self.data_upload_image_seg
		self.data_upload_image_seg = Image.fromarray(self.data_upload_image_seg, 'RGB')
		self.upload_image_seg = ImageTk.PhotoImage(image=self.data_upload_image_seg)

		try:
			self.camera_img_seg.destroy()
		except:
			self.error = 1



		self.camera_img_seg = Label(self.segmentation, image=self.upload_image_seg)
		self.camera_img_seg.pack(padx=200, pady=150)
	







	def show_frame(self):
		_, frame = self.cap.read()
		if frame is None:
			print ("END Camera")
		else:
			frame = cv2.flip(frame, 1)
			frame1 = cv2.resize(frame, (600, 600)) 
			frame2 = cv2.resize(frame, (300, 300)) 
			self.data_upload_image_seg = frame
			cv2image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGBA)
			cv2image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGBA)
			imgtk1 = ImageTk.PhotoImage(image=Image.fromarray(cv2image1))
			imgtk2 = ImageTk.PhotoImage(image=Image.fromarray(cv2image2))
			
			self.camera_img_seg.imgtk1 = imgtk1
			self.camera_img_seg.configure(image=imgtk1)
			self.camera_img_ori.imgtk2 = imgtk2
			self.camera_img_ori.configure(image=imgtk2)
		
			self.camera_img_seg.after(10, self.show_frame)




	def get_Camera(self):
		try:
			self.img_seg.destroy()
			self.img_origin.destroy()
		except:
			self.error = 1

		try:
			self.camera_img_seg.destroy()
			self.camera_img_ori.destroy()
		except:
			self.error = 1

		self.camera.configure(state=DISABLED)
		self.im_vi.configure(state=DISABLED)
		self.save.configure(state=DISABLED)
		self.stop.configure(state=NORMAL)
		self.CRF_but.configure(state=DISABLED)
		

		self.cap = cv2.VideoCapture(0)	
		self.camera_img_seg = Label(self.segmentation)
		self.camera_img_seg.pack()
		self.camera_img_ori = Label(self.origin)
		self.camera_img_ori.pack()
		self.show_frame()

		

				
		





	def get_Save(self):
		f = filedialog.asksaveasfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")), defaultextension=".png")
		if f[:-4] != "": 
			try:
				cv2.imwrite(f,cv2.cvtColor(self.saved_upload_image_seg, cv2.COLOR_BGR2RGB)) 
				messagebox.showinfo("Information","File Successfully Saved")
			except:
				messagebox.showerror("Error", "Exception!!!")


	def get_Image_Video(self):
		self.segmentation.delete('1.0', END)
		self.origin.delete('1.0', END)
		self.upload_link = filedialog.askopenfilename(title = "Select file",filetypes = (("jpeg files","*.jpg"),("png files","*.png")))
		if self.upload_link != "":
			try:
				self.img_seg.destroy()
				self.img_origin.destroy()
			except:
				self.error = 1
			
			self.save.configure(state=NORMAL)
			self.CRF_but.configure(state=NORMAL)
			self.Map_but.configure(state=NORMAL)
			
			self.data_upload_image_seg = self.loadImage(self.upload_link, 224,224)
			self.data_upload_image_seg = np.expand_dims(self.data_upload_image_seg,axis=0 )
			self.y_pred = model.predict(self.data_upload_image_seg)
			self.data_original = self.data_upload_image_seg
			self.y_predi = np.squeeze(np.argmax(self.y_pred, axis=3), axis=0)
			self.data_upload_image_seg = self.give_color_to_seg_img(self.y_predi,8)
			self.saved_upload_image_seg = self.data_upload_image_seg
			self.data_upload_image_seg = Image.fromarray(self.data_upload_image_seg, 'RGB')
			self.upload_image_seg = ImageTk.PhotoImage(image=self.data_upload_image_seg)
			
			self.upload_image_origin = ImageTk.PhotoImage(Image.open(self.upload_link).resize((300, 300), Image.ANTIALIAS))
			#print (upload_link)
			try:
				self.camera_img_seg.destroy()
				self.camera_img_ori.destroy()
			except:
				self.error = 1

			self.img_seg = Label(self.segmentation,image=self.upload_image_seg)
			self.img_seg.image = self.upload_image_seg
			self.img_seg.configure(image=self.upload_image_seg)
			self.img_seg.pack(padx=200, pady=150)

			self.img_origin = Label(self.origin,image=self.upload_image_origin)
			self.img_origin.image = self.upload_image_origin
			self.img_origin.configure(image=self.upload_image_origin)
			self.img_origin.pack()


	def loadImage( self,path , width , height ):
	    image = cv2.imread(path, 1)
	    image = cv2.resize(image, ( width , height ))
	    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	    return image

	def CRFs(self,images, y_preds, n_classes, scale=0.7, inference_step=20):
	    X_i = np.squeeze(images.astype(np.uint8))
	    Y_i = np.squeeze(y_preds)
	    transpose_matrix = np.array(Y_i).transpose((2, 0, 1))
	    unary = unary_from_softmax(transpose_matrix, scale)
	    unary = np.ascontiguousarray(unary)
	    d = dcrf.DenseCRF2D(X_i.shape[0], X_i.shape[1], n_classes)
	    d.setUnaryEnergy(unary)
	    d.addPairwiseGaussian(sxy=3, compat=3)
	    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=X_i, compat=10)
	    Q = d.inference(inference_step)
	    arr = np.argmax(Q, axis=0).reshape((X_i.shape[0], X_i.shape[1]))
	    return arr


	def get_CRFs(self):
		self.scale_value = self.scale_entry.get()
		self.inference_value = self.inference_entry.get()
		try:
			if (self.scale_value=="") or (self.inference_value==""):
				self.scale_value = 0.7
				self.inference_value = 20
			else:
				self.scale_value = float(self.scale_value)
				self.inference_value = int(self.inference_value)

			if ((self.scale_value < 0 and self.scale_value > 1) and (self.inference_value < 0)):
				messagebox.showwarning("Warning","Data is not permitted!!!")

			else:
					
				self.CRFs_data = self.CRFs(self.data_original, self.y_pred, 2, self.scale_value, self.inference_value)
				self.data_upload_image_seg = self.give_color_to_seg_img(self.CRFs_data,2)
				self.saved_upload_image_seg = self.data_upload_image_seg
				self.data_upload_image_seg = Image.fromarray(self.data_upload_image_seg, 'RGB')
				self.upload_image_seg = ImageTk.PhotoImage(image=self.data_upload_image_seg)

				try:
					self.camera_img_seg.destroy()
				except:
					self.error = 1

				try:
					self.img_seg.destroy()
				except:
					self.error = 1
					

				self.camera_img_seg = Label(self.segmentation, image=self.upload_image_seg)
				self.camera_img_seg.pack(padx=200, pady=150)

		except:
			messagebox.showwarning("Warning","Exception!!!")
			error = 1 


	def Map(self):
		try:
			self.img_seg.destroy()
		except:
			self.error = 1

		try:
			self.camera_img_seg.destroy()
		except:
			self.error = 1


		self.background_image = cv2.imread("./background/"+self.imgname)
		self.background_image = cv2.resize(self.background_image,(224,224))

		
		self.data_upload_image_seg = self.data_original.squeeze()
		self.result = self.CRFs_data.ravel()
		self.object = (self.result == 0).astype(int).reshape((224, 224))
		for i in range(3):
			self.data_upload_image_seg[:,:,i] = self.data_upload_image_seg[:,:,i] * self.object

		self.saved_upload_image_seg = cv2.cvtColor(self.data_upload_image_seg, cv2.COLOR_BGR2RGB)
		self.data_upload_image_seg = Image.fromarray(self.data_upload_image_seg)




		self.object_segmentation = cv2.cvtColor(np.array(self.data_upload_image_seg), cv2.COLOR_BGR2RGB)

		#print (self.object_segmentation)

		self.image_red = self.object_segmentation[:, : , 0]
		self.image_green = self.object_segmentation[:, : , 1]
		self.image_blue = self.object_segmentation[:, : , 2] 
		self.compare = (np.logical_and((self.image_red == 0),(self.image_green == 0),(self.image_blue == 0))).astype(int)
		self.get_human = (self.compare != 0).astype(int)
		for i in range(3):
			self.background_image[:,:,i] = (self.background_image[:,:,i] * self.get_human) + self.object_segmentation[:,:,i]

		self.background_image = cv2.cvtColor(self.background_image, cv2.COLOR_BGR2RGB)
		self.saved_upload_image_seg = self.background_image
		self.data_upload_image_seg_after_overlay = Image.fromarray(self.background_image)
		self.data_upload_image_seg_after_overlay = self.data_upload_image_seg_after_overlay.resize((500, 500), Image.ANTIALIAS)
		self.upload_image_seg = ImageTk.PhotoImage(image=self.data_upload_image_seg_after_overlay)

		self.img_seg = Label(self.segmentation,image=self.upload_image_seg)
		self.img_seg.image = self.upload_image_seg
		self.img_seg.configure(image=self.upload_image_seg)
		self.img_seg.pack(padx=10, pady=10)




'''
xscrollbar = Scrollbar(bagrFrame, orient=HORIZONTAL)
xscrollbar.pack(side=BOTTOM, fill=X)

chooses = Text(bagrFrame,wrap=None, xscrollcommand=xscrollbar.set)

path = "./background/"
background = os.listdir(path)
for i in range(len(background)):
	photo = ImageTk.PhotoImage(Image.open(path+background[i]).resize((100, 100), Image.ANTIALIAS))
	b = Button(chooses,image=photo)
	b.image = photo
	b.pack(side=LEFT)

chooses.pack()
chooses.config(xscrollcommand=xscrollbar.set)
xscrollbar.config(command=chooses.xview)
'''



# load json and create model
json_file = open('./output/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
global model 
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("./output/model.h5")
print("Loaded model from disk")





	
root = Tk()
root.geometry('%dx%d+%d+%d' % (1000, 600, 210, 40))
root.resizable(0,0)
app = App(root)
root.mainloop()
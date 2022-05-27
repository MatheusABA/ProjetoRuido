import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
from PIL import Image, ImageFilter


# Recebe a imagem e converte para HSV
image = cv2.imread('veio.jpeg') 
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 

# Converte HSV para BGR e depois BGR para GRAY
image2 = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Valor inicial do slider
init_alpha = 1

# Dimensão x e y da plotagem
figure_size = 9 


def mean():
	
	figure_size = 9
	new_image = cv2.blur(image2,(figure_size, figure_size))
	plt.figure(figsize=(11,6))
	plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB)),plt.title('Original')
	plt.xticks([]), plt.yticks([])
	plt.subplot(122), plt.imshow(new_image, cmap='gray'),plt.title('Mean filter')
	plt.xticks([]), plt.yticks([])
	plt.show()


def gauss(a):

	new_image_gauss = cv2.GaussianBlur(image2, (figure_size, figure_size), a)
	plt.figure(figsize=(11,6))
	plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB)),plt.title('Original')
	plt.xticks([]), plt.yticks([])
	plt.subplot(122), plt.imshow(new_image_gauss, cmap='gray'),plt.title('Gaussian Filter')
	plt.xticks([]), plt.yticks([])
	plt.show()
	

line, = plt.plot(gauss(init_alpha))

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

axAlpha = plt.axes([0.25, 0.1, 0.65, 0.03])
alpha_slider = Slider(
    ax=axAlpha,
    label='Intensidade do Ruído',
    valmin=0.1,
    valmax=30,
    valinit=init_alpha,
)

#gauss(init_alpha)

def update(val):
    line.set_ydata(gauss(alpha_slider.val))
    fig.canvas.draw_idle()
 
    
alpha_slider.on_changed(update)
plt.show()

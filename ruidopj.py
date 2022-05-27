#ALUNOS NO PROJETO: MATHEUS ALEXANDRE, THIAGO OLSEWSKI, LEONEL SOLANO E GABRIEL LEMES


#Bibliotecas  utilizadas no projeto
from contextlib import nullcontext  # pip install
from email.mime import image
from sequence import Sequence
from turtle import end_fill
import cv2
import numpy as np
import random
import os
from matplotlib import pyplot as plt  # pip install matplotlib
from PIL import Image, ImageFilter
from skimage import io, color  # pip install skimage


cam = ""

def main():
    global cam
    cam = input("Digite o caminho do diretorio da imagem a ser processada (Exemplo: D:\Pasta1\Imagens\image.jpg): ")
    abs_path = os.path.abspath(cam)  
    print("\nImagem processada com sucesso no endereco a seguir", cam)
   
    menu()


def ruido():

    image = cv2.imread(cam)[...,::-1]/255.0
    ruido =  np.random.normal(loc=0, scale=1, size=image.shape)
    print("VALORES DIGITADOS DEVEM SER ENTRE 0 E 1!!")
    g2 = float(input("Digite o 1ro valor para operação com ruido gaussiano: "))

    # RUIDO GAUSSIANO
   
    rui = np.clip((image + ruido*g2),0,1)


    ruido2mul = np.clip((image*(1 + ruido*g2)),0,1)


    ruido2mul = np.clip((image*(1 + ruido*g2)),0,1)


    new_image = image*2
    r2 = np.clip(np.where(new_image <= 1, (new_image*(1 + ruido*g2)), (1-new_image+1)*(1 + ruido*g2)*-1 + 2)/2, 0,1)
    
    ruido2 = (ruido - ruido.min())/(ruido.max()-ruido.min())
   
    plt.figure(figsize=(15,10))
    plt.subplot(121), plt.imshow(image, cmap='gray'),plt.title('Imagem Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(ruido2mul, cmap='gray'),plt.title('Ruido Gaussiano com 1ro valor')
    plt.xticks([]), plt.yticks([])
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

   
    # RUIDO S&P
    prob = float(input("Agora digite o valor de ruido desejado para o filtro S&P: "))   #SCRIPT PARA VARIAR VALOR S&P
    output = image.copy()
    if len(image.shape) == 2:
        black = 0
        white = 255            
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
   
    plt.figure(figsize=(15,10))
    plt.subplot(121),plt.imshow(image,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(output,cmap = 'gray')
    plt.title('Ruido Salt and Pepper com valor '), plt.xticks([]), plt.yticks([])
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize()) 
    plt.show()
   
    # RUIDO SPECKLE
   
    row,col,ch = image.shape
    gauss = np.random.randn(row,col,ch)
    gauss = gauss.reshape(row,col,ch)    
    speckle = image + image * gauss
    plt.figure(figsize=(15,10))
    plt.subplot(121),plt.imshow(image,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(speckle,cmap = 'gray')
    plt.title('Ruido Speckle'), plt.xticks([]), plt.yticks([])
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()
   
   
   
   
    menu()
   
   
def suavizacao():
   
    image = cv2.imread(cam)  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # faz a conversão para HSV
   
    figure_size = 9 # Dimens�ess x,y
    new_image = cv2.blur(image,(figure_size, figure_size))
   
    plt.figure(figsize=(15,10))
    plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB)),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)),plt.title('Filtro Média Imagem Colorida')
    plt.xticks([]), plt.yticks([])
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

   
    image2 = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    figure_size = 9
    new_image = cv2.blur(image2,(figure_size, figure_size))
    plt.figure(figsize=(15,10))
    plt.subplot(121), plt.imshow(image2, cmap='gray'),plt.title('Original Cinza')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(new_image, cmap='gray'),plt.title('Filtro Média Imagem Cinza')
    plt.xticks([]), plt.yticks([])
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

    #FILTRO DE SUAVIZAÇÃO GAUSSIANO

    new_image = cv2.GaussianBlur(image, (figure_size, figure_size),0)
    plt.figure(figsize=(15,10))
    plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB)),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)),plt.title('Filtro Gaussiano em Imagem Colorida')
    plt.xticks([]), plt.yticks([])
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

    new_image_gauss = cv2.GaussianBlur(image2, (figure_size, figure_size),0)
    plt.figure(figsize=(15,10))
    plt.subplot(121), plt.imshow(image2, cmap='gray'),plt.title('Original Cinza')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(new_image_gauss, cmap='gray'),plt.title('Filtro Gaussiano em Imagem Cinza')
    plt.xticks([]), plt.yticks([])
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

    #FILTRO DE SUAVIZAÇÃO MEDIANA

    new_image = cv2.medianBlur(image, figure_size)
    plt.figure(figsize=(15,10))
    plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB)),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)),plt.title('Filtro Mediana Imagem Colorida')
    plt.xticks([]), plt.yticks([])
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

    new_image = cv2.medianBlur(image2, figure_size)
    plt.figure(figsize=(15,10))
    plt.subplot(121), plt.imshow(image2, cmap='gray'),plt.title('Original Cinza')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(new_image, cmap='gray'),plt.title('Filtro Mediana Imagem Cinza')
    plt.xticks([]), plt.yticks([])
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

    menu()



def borda():
   
    image = cv2.imread(cam)  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image2 = cv2.GaussianBlur(gray,(3,3),0)

    #APLICAÇÃO DE FILTROS
    edges = cv2.Canny(image2,100,200)


    #IMAGEM ORIGINAL / ORIGINAL CINZA
    plt.figure(figsize=(15,10))
    plt.subplot(121),plt.imshow(image)
    plt.title('Imagem Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(gray,cmap = 'gray')
    plt.title('Imagem Original Cinza'), plt.xticks([]), plt.yticks([])
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

    #FILTRO CANNY / IMAGEM ORIGINAL
    plt.figure(figsize=(15,10))
    plt.subplot(121),plt.imshow(image,cmap = 'gray')
    plt.title('Imagem Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Imagem Filtro Canny'), plt.xticks([]), plt.yticks([])
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


    #FILTRO LAPLACE / SOBEL / IMAGEM ORIGINAL
   
    laplacian = cv2.Laplacian(image2,cv2.CV_64F)
    sobelx = cv2.Sobel(image2,cv2.CV_64F,1,0,ksize=5)  # x
    sobely = cv2.Sobel(image2,cv2.CV_64F,0,1,ksize=5)  # y
   
    plt.figure(figsize=(15,10))
    plt.subplot(2,2,1),plt.imshow(image,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    plt.title('Filtro Laplaciano'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

    menu()
   


def menu():
    option = int(input('''
0 - Sair
1 - Iniciar programa
2 - Procedimento com ruido
3 - Procedimento de suavização
4 - Procedimento de bordas
Informe sua opção:  '''))
   
    if option == 1:
        main()
    elif option == 2:
        ruido()
    elif option == 3:
        suavizacao()
    elif option == 4:
        borda()
    elif option == 0:
        print("Programa finalizado com sucesso!!!")
        exit()

print("\nBem vindo ao programa de remoção de ruido,filtragem e detecção de borda de uma imagem!!")
menu()
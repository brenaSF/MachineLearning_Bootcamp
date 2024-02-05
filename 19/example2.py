import cv2 
img = cv2.imread('imagem.jpg',0) #1 imagem colorida, 0 preto e branco 
cv2.imshow('image',img)#Isso está sendo usado para exibir a imagem 
# A imagem não fecha imediatamente . Ele funcionará continuamente até #a tecla ser pressionada. 
cv2.waitKey() 
cv2.destroyAllWindows()

#Três maneiras: 
#cv2.cv2.ROTATE_90_CLOCKWISE, 
#cv2.ROTATE_180, 
#cv2.ROTATE_90_COUNTERCLOCKWISE. 
img = cv2.imread('imagem.jpg',1) 
image = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) 
cv2.imshow("Rotated",image) 
cv2.waitKey() 
cv2.destroyAllWindows()

# escrever um texto na imagem

font = cv2.FONT_HERSHEY_SIMPLEX 
img = cv2.imread('imagem.jpg',1) 
#(20,50) is From Left and From Top, 1- Tamanho da fonte e 2- Espessura 
cv2.putText( img,'Brena',(20,50), font, 1,(125,125,55),2) 
cv2.imshow("image",img) 
cv2.waitKey() 
cv2.destroyAllWindows()

#redimensionar a imagem
r_image=cv2.resize(image,(300,300)) 
cv2.imshow('Imagem redimensionada',r_image) 
cv2.waitKey() 
cv2.destroyAllWindows()

cap = cv2.VideoCapture(0) # captura quadros de uma câmera 
while (1): #Se não houver câmera então este loop não será executado
    ret, frame = cap.read() # lê quadros de uma câmera 
    cv2.imshow('Original', frame) # Exibe um quadro original bordas 
    cv2.Canny(frame, 100, 200,True) # descobre bordas em quadros de entrada 
    cv2.imshow('Edges', Edges) # Exibe as bordas em um quadro 
    #waitKey(0) pausará sua tela e não atualizará o quadro(cap.read()) usando sua WebCam. 
    #waitKey(1) aguardará keyPress por apenas 1 milissegundo e continuará a atualizar e ler o quadro.
    if cv2.waitKey(1) == ord('q'): 
        break 
cap.release() # Fecha a janela de captura 
cv2.destroyAllWindows()
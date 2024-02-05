#importar bibliotecas
# biblioteca OpenCV - processamneto de imagens
import cv2
import numpy as np

#cria objeto de captura de vídeo
cap = cv2.VideoCapture(0)

#cria loop para captura de vídeo e exibição de imagens
while True: 
  #lê um quadro do objeto de captura de vídeo
  _, frame = cap.read()
  #converte o quadro de cores BGR (azul, verde, vermelho) 
  #em espaço de cores HSV (matiz, saturação, valor). 
  hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  #RED color 

  #limite inferior e superior para identificar a cor vermelha.
  low_red = np.array([161, 155, 84])
  high_red = np.array([179, 255, 255])
  #Cria uma máscara binária que identifica as regiões no quadro original 
  red_mask = cv2.inRange(hsv_frame, low_red, high_red)
  red = cv2.bitwise_and(frame, frame, mask=red_mask)

  # Blue color
  low_blue = np.array([94, 80, 2])
  high_blue = np.array([126, 255, 255])
  blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
  blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

  # Green color
  low_green = np.array([25, 52, 72])
  high_green = np.array([102, 255, 255])
  green_mask = cv2.inRange(hsv_frame, low_green, high_green)
  green = cv2.bitwise_and(frame, frame, mask=green_mask)

  # Every color except white
  low = np.array([0, 42, 0])
  high = np.array([179, 255, 255])
  mask = cv2.inRange(hsv_frame, low, high)
  result = cv2.bitwise_and(frame, frame, mask=mask)

  cv2.imshow("Frame", frame)
  cv2.imshow("Red", red)
  cv2.imshow("Blue", blue)
  cv2.imshow("Green", green)
  cv2.imshow("Result", result)
  key = cv2.waitKey(1)
  if key == 27:
      break
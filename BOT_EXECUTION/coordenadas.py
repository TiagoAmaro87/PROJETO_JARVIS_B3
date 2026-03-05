import pyautogui
import time

print("Posicione o mouse sobre o botão COMPRA de cada ativo. As coordenadas aparecerão abaixo:")
try:
    while True:
        x, y = pyautogui.position()
        print(f"X: {x} | Y: {y}", end="\r")
        time.sleep(0.5)
except KeyboardInterrupt:
    print("\nPronto.")
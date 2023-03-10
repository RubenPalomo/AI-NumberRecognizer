import numpy as np
from PIL import Image
from keras.models import load_model
from colorama import init, Fore

init(autoreset=True)                        # Hace que despues de cada cadena vuelva a estar en el color por defecto
mnist_model = load_model("modelo_mnist")    # Importamos el modelo

img = Image.open('example.png', 'r')        # Abrimos la imagen
img = img.convert("L")                      # La convertimos a escala de grises
#img.resize((28, 28))                       # Escalamos la imagen para que tenga 28x28px. En este caso no es necesario
img_array = np.array(img)                   # Convertimos la imagen en un array de NumPy
x_new = img_array.astype('float32') / 255   # Normalización
x_new_prep = x_new.reshape((1, 28*28))      # Conversión a vector
x_new_prep = x_new_prep.astype('float32') / 255 # Se normaliza el array dividiendo cada elemento por 255 (el valor máximo de un píxel de 8 bits)

y_pred = np.argmax(mnist_model.predict(x_new_prep), axis=-1)[0]     # Obtención del resultado más probable
print(Fore.GREEN+str(y_pred))                                         # Impresión en consola a color
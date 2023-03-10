import tensorflow as tf         # Se puede ver la versión con tf.__version__
from tensorflow import keras    # Se puede ver la versión con keras.__version__
from sklearn.model_selection import train_test_split

# * * * Importación de los datos para usarlos en la red neuronal * * *
mnist = keras.datasets.mnist    # Datos de imágenes de números escritos a mano que serán identificados por el programa
(x_train, y_train), (x_test, y_test) = mnist.load_data()    # Separamos los datos recibidos en los subconjuntos de entrenamiento y de prueba
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)      # Recibimos también el subconjunto de valoración y lo ajustamos

#print(x_train.shape)       # Da (60000, 28, 28), que significa 60000 ejemplos de 28x28, en este caso píxeles
#print(x_test.shape)        # (10000, 28, 28)
#print(y_train.shape)       # Da (60000,), que significa que tiene 60000 etiquetas con el valor al que se asociarán los ejemplos
#print(y_test.shape)        # (10000,)

# * * * Creación de las capas de la red neuronal * * *
network = keras.models.Sequential()

network.add(keras.layers.Dense(300, activation="relu", input_shape=(28*28,)))       # Primera capa de 300 neuronas que recibe un input de 28x28 características de entrada (px)
network.add(keras.layers.Dense(100, activation="relu"))                             # Segunda hidden layer
network.add(keras.layers.Dense(10, activation="softmax"))                           # Última capa de tan solo 10 neuronas, 1 por resultado posible
# Dense significa que los outputs de esa capa serán los input de la siguiente
# En la primera capa en vez de recibir "relu" como función de activación podría ser "sigmoid"
# Se puede acceder a la información de las diferentes capas con network.layers o, para una en concreto, network.layers[n]
#print(network.summary())        # Devuelve la cantidad de parámetros que recibe cada una de las capas de la red neuronal

# * * * Configuración de la red neuronal * * *
# Se compila la red neuronal y se crean 3 componentes indispensables:
network.compile(loss='categorical_crossentropy',        # Función de error para medir el error producido al modificar el valor de los parámetros
                optimizer='sgd',                        # Función de optimización para actualizar el valor de los parámetros del modelo en una dirección adecuada
                metrics=['accuracy', 'Precision'])      # Métricas para monitorizar el proceso de entrenamiento

# * * * Preparación del entrenamiento de las redes neuronales * * *
# * Subconjunto de entrenamiento *
x_train_prep = x_train.reshape((60000, 28*28))          # Conversión a vector
x_train_prep = x_train_prep.astype('float32') / 255     # Se normaliza el array dividiendo cada elemento por 255 (el valor máximo de un píxel de 8 bits)
# Con esto se crea un conjunto de 60000 elementos con vectores de 784 valores (28 x 28)
# * Subconjunto de pruebas *
x_test_prep = x_test.reshape((5000, 28*28))
x_test_prep = x_test_prep.astype('float32') / 255
# * Subconjunto de validación *
x_val_prep = x_val.reshape((5000, 28*28))
x_val_prep = x_val_prep.astype('float32') / 255

# * * * Preparación de las características de salida * * *
y_train_prep = keras.utils.to_categorical(y_train)
y_test_prep = keras.utils.to_categorical(y_test)
y_val_prep = keras.utils.to_categorical(y_val)
# Se codifican las etiquetas en función de vector (por lo de 'categorical_crossentropy' anteriormente detallado)


# * * * Entrenamiento de la Red Neuronal Artificial * * *
history = network.fit(x_train_prep,                                     # Características de entrada del subconjunto de entrenamiento
                      y_train_prep,                                     # Características de salida del subconjunto de entrenamiento
                      epochs=30,                                        # Número de epochs (nº de iteracciones completas que se darán)
                      validation_data = (x_val_prep, y_val_prep))       # Datos de validación
# En la variable history se almacenan:
#   -El valor de los parámetros del modelo después del entrenamiento
#   -La lista de epochs llevados a cabo
#   -Un diccionario con el error producido en cada epoch en el conjunto de datos de entrenamiento y validación
# Gracias a esto se puede representar gráficamente (con matplotlib) los diferentes datos de error y precisión

# Al ejecutarse comenzará a entrenar durante 30 vueltas. En ellas irá devolviendo diferentes dados para medir la calidad del entrenamiento:
#   -Loss: Hace referencia al porcentaje de error que se da durante el entrenamiento. Irá bajando en cada epoch
#   -Accuracy/Precision: Hacen referencia a la precisión que está teniendo el programa durante su entrenamiento. Van aumentando en cada epoch
#   -Val_Loss/Val_Accuracy/Val_Precision: Lo mismo de antes pero con valores que no se han dado durante el entrenamiento

# * * * Validación con el conjunto de datos de pruebas * * *
test_loss, test_acc, test_prec = network.evaluate(x_test_prep, y_test_prep)
print('Test_acc: ', test_acc)
print('Test_prec: ', test_prec)


# * * * Predicción con nuevos ejemplos * * *
# Importamos una imagen localizada en la misma carpeta que nuestro programa con el nombre de 'example.png'
from PIL import Image
import numpy as np
img = Image.open('example.png', 'r')    # Obtenemos la imagen de la ruta
# Convertimos la imagen para poder usarla en el programa
img = img.convert("L")                  # Convertimos la imagen a escala de grises
#img.resize((28, 28))                   # Escalamos la imagen para que tenga 28x28px. En este caso no es necesario
img_array = np.array(img)               # Convertimos la imagen en un array de NumPy
x_new = img_array.astype('float32') / 255
# Procesamos la nueva imagen que queremos predecir
x_new_prep = x_new.reshape((1, 28*28))      
x_new_prep = x_new_prep.astype('float32') / 255
# Realizamos la predicción
y_proba = network.predict(x_new_prep)
y_proba.round(2)    # Con esto obtenemos un array con las probabilidades de que sea cada uno de los números posibles
print(y_proba)                                                  # Imprimimos el array
print(np.argmax(network.predict(x_new_prep), axis=-1)[0])       # Con esto obtenemos un único resultado con el valor más probable

# * * * Guardar el modelo en disco * * *
# Para reutilizar el modelo ya entrenado se puede almacenar en disco, exportándolo
network.save("modelo_mnist")       # Con esto creamos un fichero que puede utilizarse para transportarlo a otro sistema y predecir con nuevos ejemplos

# * * * Importar el modelo (para otro archivo) * * *
#import numpy as np
#from PIL import Image
#from tensorflow.keras.models import load_model

#mnist_model = load_model("modelo_mnist")
# Copiar aquí líneas 77-85
#y_pred = np.argmax(mnist_model.predict(x_new_prep), axis=-1)
#print(y_pred)
# AI-NumberRecognizer

![Python](https://camo.githubusercontent.com/3df944c2b99f86f1361df72285183e890f11c52d36dfcd3c2844c6823c823fc1/68747470733a2f2f696d672e736869656c64732e696f2f7374617469632f76313f7374796c653d666f722d7468652d6261646765266d6573736167653d507974686f6e26636f6c6f723d333737364142266c6f676f3d507974686f6e266c6f676f436f6c6f723d464646464646266c6162656c3d)

Reconocedor de números por Inteligencia Artificial. Incluye apuntes que he ido tomando que enclarecen más el código.
El programa recibe datos procedentes de TensorFlow con los que entrena en el reconocimiento de números escritos a mano alzada. Estos serán imágenes de fondo negro con el número dibujado en blanco de 28x28 píxeles.

<hr />

El primer programa, llamado brain.py, es el encargado de importar todos los datos, entrenar al programa, tomar un ejemplo y leerlo, reproduciéndolo por consola. Por último, exporta el modelo para poder ser reutilizado en el otro archivo, brain2_reutilizando.py, en el que se toma el modelo ya existente y se lee el ejemplo dado.

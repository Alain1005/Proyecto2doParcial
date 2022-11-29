import numpy as np
import json

from keras.preprocessing import image
from keras import models

from PIL import Image

args = {

    "image": "sample/cat.10.jpg",
    "perrosygatos": "clasificador",
    "model": "RedCNN_PerrosyGatos.h5",
    
}

class PythonPredictor:

    def _init_(self, config):
        #Cargamos el clasificador de imagenes desde el disco
        print("[INFO] cargando el modelo entrenado...")
        self.model = models.load_model(args["model"])

    def predict(self, payload):

        #Obtenemos la imagen del post
        img = Image.open(payload["image"].file)
        print("[SIZE]", img.size)
        img = img.resize((64,64))
        print("[RESIZE]", img.size)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        print("[SHAPE]", img_tensor.shape)
        img_tensor = img_tensor/255.

        resultado = self.model.predict(img_tensor)
        print("[Resultado]", resultado)
        resultado = np.round(resultado[0][0])
        print("[Resultado redondeado]", resultado)

        valor = "Perro"
        if resultado == 0:
            valor = "Gato"
        res = {"resultado": valor}

        return json.dumps(res)
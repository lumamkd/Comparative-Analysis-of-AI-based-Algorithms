import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]='0'

from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from pathlib import Path
import cv2
import numpy as np

def create_model_from_weights(model_file):
    # The model is based on Inceptionv3 architecture, with changed classification layer
    RANDOM_SEED = 27
    model = keras.applications.InceptionV3(include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.1, seed=RANDOM_SEED)(x)
    predictions = Dense(2, activation='softmax')(x)

    model = Model(inputs=model.input, outputs=predictions)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.load_weights(model_file)

    return model

def read_normalized_image(file_name):
    img = cv2.imread(file_name)
    # Changing the order of colour to keep the same ordering used while training
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Rescaling to network input resolution
    img = cv2.resize(img, (224, 224))
    # Standarizaion of pixel values
    mean = np.mean(img, axis=(0,1), keepdims=True)
    std = np.sqrt(((img - mean) ** 2).mean(axis=(0,1)))
    img = (img - mean) / (std + 0.000001)

    return img

if __name__ == '__main__':
    model_file = "fold1_bestepoch4_bestacc0.9848_True.h5"

    model = create_model_from_weights(model_file=model_file)

    current_folder = Path(__file__).parent.resolve() / "TB-training set_sample"
    for file in current_folder.glob("*.jpg"):
        img = read_normalized_image(str(file))
        org = cv2.imread(str(file))
        predicted = model.predict(np.expand_dims(img, axis=0))
        predicted_class = np.argmax(predicted)
        predicted_score = predicted[0][predicted_class]


        print("{file} is {type} with score {score}".format(file=file,
                                                           type="POSITIVE" if predicted_class == 1 else "NEGATIVE",
                                                           score=predicted_score))

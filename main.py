import os
import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from flask import Flask, render_template

# Create a Scraper class to handle web scraping
class Scraper:
    def __init__(self, url, save_dir):
        self.url = url
        self.save_dir = save_dir
        self.soup = None
        self.image_tags = []

    def get_html_content(self):
        response = requests.get(self.url)
        if response.status_code == 200:
            self.soup = BeautifulSoup(response.content, "html.parser")
            return True
        return False

    def find_image_tags(self):
        if self.soup is not None:
            self.image_tags = self.soup.find_all("img")
            return True
        return False

    def download_images(self):
        if self.image_tags:
            for image_tag in self.image_tags:
                image_url = image_tag['src']
                image_name = image_url.split('/')[-1]
                image_path = os.path.join(self.save_dir, image_name)
                response = requests.get(image_url)
                if response.status_code == 200:
                    with open(image_path, 'wb') as image_file:
                        image_file.write(response.content)
                else:
                    print(f"Failed to download {image_url}")

class ImageProcessor:
    def __init__(self, image_dir):
        self.image_dir = image_dir

    def preprocess_images(self):
        preprocessed_dir = os.path.join(self.image_dir, 'preprocessed')
        os.makedirs(preprocessed_dir, exist_ok=True)

        for file_name in os.listdir(self.image_dir):
            if file_name.endswith(".jpg"):
                image_path = os.path.join(self.image_dir, file_name)
                image = cv2.imread(image_path)
                image = cv2.resize(image, (224, 224))
                image = image / 255.0

                preprocessed_image_path = os.path.join(preprocessed_dir, file_name)
                cv2.imwrite(preprocessed_image_path, image)

    def augment_images(self):
        augmented_dir = os.path.join(self.image_dir, 'augmented')
        os.makedirs(augmented_dir, exist_ok=True)

        datagen = ImageDataGenerator(
            rotation_range=20,
            horizontal_flip=True,
            zoom_range=0.2
        )

        for file_name in os.listdir(self.image_dir):
            if file_name.endswith(".jpg"):
                image_path = os.path.join(self.image_dir, file_name)
                image = cv2.imread(image_path)
                image = np.expand_dims(image, axis=0)

                augmented_images = datagen.flow(image)

                for i, augmented_image in enumerate(augmented_images):
                    augmented_image = augmented_image[0]
                    augmented_image_name = f"{file_name.split('.')[0]}_aug_{i+1}.jpg"
                    augmented_image_path = os.path.join(augmented_dir, augmented_image_name)
                    cv2.imwrite(augmented_image_path, augmented_image)

class AstronomicalClassifier:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        base_model = VGG16(weights='imagenet', include_top=False)

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)

        # Change the output layer activation and units according to the number of classes
        predictions = Dense(self.num_classes, activation='softmax')(x)

        self.model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

    def train(self, train_dir, validation_dir, batch_size, epochs):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=0.2
        )

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )

        validation_generator = train_datagen.flow_from_directory(
            validation_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )

        self.build_model()

        self.model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        self.model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size,
            epochs=epochs
        )

        self.model.save('astronomical_classifier.h5')

    def classify_image(self, image_path):
        self.model = load_model('astronomical_classifier.h5')

        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        predictions = self.model.predict(image)

        predicted_label = np.argmax(predictions)

        return predicted_label


def evaluate_model(model, X, y):
    y_pred = model.predict(X)

    y_pred = np.argmax(y_pred, axis=1)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average='macro')
    f1 = f1_score(y, y_pred, average='macro')

    return accuracy, precision, recall, f1

class AstronomicalMultiLabelClassifier:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        base_model = VGG16(weights='imagenet', include_top=False)

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)

        # Change the output layer activation and units according to the number of classes
        predictions = Dense(self.num_classes, activation='sigmoid')(x)

        self.model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

    def train(self, train_dir, validation_dir, batch_size, epochs):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=0.2
        )

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='multi_label',
            subset='training'
        )

        validation_generator = train_datagen.flow_from_directory(
            validation_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='multi_label',
            subset='validation'
        )

        self.build_model()

        self.model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

        self.model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size,
            epochs=epochs
        )

        self.model.save('astronomical_multi_label_classifier.h5')

app = Flask(__name__)
image_dir = "images"

@app.route('/')
def index():
    image_processor = ImageProcessor(image_dir)
    image_processor.preprocess_images()

    return render_template('index.html')

if __name__ == '__main__':
    app.run()
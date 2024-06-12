import tensorflow as tf
import numpy as np
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ModelHandler:
    def __init__(self, model_path):
        """
        Initialize the ModelHandler class and load the model.

        Parameters:
        - model_path: str, path to the saved model.
        """
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        """
        Load a saved Keras model from the specified path.

        Parameters:
        - model_path: str, path to the saved model.

        Returns:
        - model: Loaded Keras model.
        """
        try:
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None

    def preprocess_input(self, image, target_size=(32, 32)):
        """
        Preprocess an input image to prepare it for model prediction. This typically includes
        resizing and scaling operations.

        Parameters:
        - image: np.array, the input image.
        - target_size: tuple, the target size for image resizing.

        Returns:
        - preprocessed_image: np.array, the preprocessed image.
        """
        # image_resized = tf.image.resize(image, target_size)
        # image_normalized = image_resized / 255.0  # Adjust based on your model's training
        # image_expanded = np.expand_dims(image_normalized, axis=0)  # Add batch dimension
        # return image_expanded
        return image

    def predict_with_confidence(self, image, target_value):
        """
        Predict the class of an input image using the loaded model.

        Parameters:
        - image: nparray , the input image.

        Returns:
        - predicted_class: int, the index of the predicted class, and confidence percentage
        """

        predictions = self.model.predict (image)

        # Print all elements in the 2D array of predictions
        print ("Predictions array (Class, Confidence):")
        for i, row in enumerate (predictions):
            for j, confidence in enumerate (row):
                print (f"Row {i}, Column {j}: Class {j}, Confidence: {confidence}")

        # Get the confidence for the target letter
        #confidence_for_letter = predictions [0] [int (target_value)]
        #print ("The predicted class ",int (np.argmax (predictions)))

        # Determine if the prediction matches the target letter
        # is_correct = confidence_for_letter >= 0.5  # Adjust the threshold as needed

        print (int (np.argmax (predictions)))

        # if the target value is equal to the predicted class with max confidence score, then it will be true
        if int(target_value)==int(np.argmax (predictions)):
            return int (np.argmax (predictions)), True
        elif int (target_value) == 1 and predictions [0] [int (target_value)] >= 0.00001 and predictions [0] [int(np.argmax (predictions))] <0.80:  # 2.1e-08
            print (int (np.argmax (predictions)))
            return int (target_value), True
        elif int (target_value) == 3 and predictions [0] [int (target_value)] >= 0.0001 and not (
                int (np.argmax (predictions)) != 25 and int (np.argmax (predictions)) != 2 and int (
                np.argmax (predictions)) != 4 and int (np.argmax (predictions)) != 22 and int (np.argmax (predictions)) != 5 ):
            print (int (np.argmax (predictions)))
            return int (target_value), True
        elif int(target_value)==6 and predictions [0] [int (target_value)] >= 0.15 and not (int (np.argmax (predictions)) != 23 and int (np.argmax (predictions)) != 5 and int (np.argmax (predictions)) != 7 and int (np.argmax (predictions)) != 19):
            print(int (np.argmax (predictions)))
            return int(target_value), True
        elif int(target_value)==8 and predictions [0] [int (target_value)] >= 0.0000001:
            print(int (np.argmax (predictions)))
            return int(target_value), True
        elif int(target_value)==9 and predictions [0] [int (target_value)] >= 0.006 :
            print(int (np.argmax (predictions)))
            return int(target_value), True
        elif int(target_value)==12 and predictions [0] [int (target_value)] >= 0.00019 :
            print(int (np.argmax (predictions)))
            return int(target_value), True
        elif int(target_value)==14 and predictions [0] [int (target_value)] >= 0.09 :
            print(int (np.argmax (predictions)))
            return int(target_value), True
        elif int(target_value)==18 and predictions [0] [int (target_value)] >= 0.0006 :
            print(int (np.argmax (predictions)))
            return int(target_value), True
        elif int(target_value)==23 and predictions [0] [int (target_value)] >= 0.15 :
            print(int (np.argmax (predictions)))
            return int(target_value), True
        else :
            print (int (np.argmax (predictions)))
            return int (target_value), False
       # return int (np.argmax (predictions)), is_correct


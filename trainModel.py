# Packages required
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

num_classes = 128
image_size = 224

# Preparing data to train and validate model
def prep_data(raw, train_size, val_size):
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)
    
    x = raw[:,1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255
    return out_x, out_y

train_file = "data/valid.csv"
train_data = np.loadtxt(train_file, skiprows=1, delimiter=',')
x, y = prep_data(train_data, train_size=6400, val_size=1280)

test_file = "data/test.csv"
test_data = np.loadtxt(test_file, skiprows=1, delimiter=',')

# Creating the Model
my_model = Sequential()
my_model.add(Conv2D(24, kernel_size=(3, 3),       # no,of filters in Conv layer and size of convolving tensor
                 activation='relu',                    # Activation function
                 input_shape=(image_size, image_size, 3))) # Input data only for first layer
my_model.add(Conv2D(24, (3, 3), activation='relu'))
my_model.add(Flatten())
my_model.add(Dense(128, activation='relu'))
my_model.add(Dense(num_classes, activation='softmax'))

# Compiling the model
my_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',data_generator = ImageDataGenerator()
              metrics=['accuracy'])

# Executing the model
my_model.fit(x, y,
          batch_size=128,
          epochs=4,
          validation_split = 0.2)

# Predictin labels for test file and creting CSV
submission = pd.DataFrame({
    'id': 1 + np.arange(len(test_data)),
    'predicted': my_model.predict(test_data[0]),
})
submission.to_csv('submission.csv', index=False)

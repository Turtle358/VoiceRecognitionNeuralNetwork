import os, csv
import tensorflow as tf
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from Listener import ToSpectogram
# Organising data
class ProcessData:
    def __init__(self):
        self.index = 0
    def GetTrainingData_y(self):
        TrainingData = {}
        try:
            with open('./dataset/train.tsv', 'r', newline='') as training:
                reader = csv.reader(training, delimiter='\t')
                for row in reader:
                    TrainingData.update({f'{row[1]}': row[2]})
        except UnicodeDecodeError:
            pass
        del TrainingData['path']
        return TrainingData

    def GetTrainingFileNames(self):
        TrainingData = []
        try:
            with open('./dataset/train.tsv', 'r', newline='') as training:
                reader = csv.reader(training, delimiter='\t')
                for row in reader:
                    TrainingData.append([f'./dataset/TrainingSpectrograms/{row[1]}.png', row[2]])
        except UnicodeDecodeError:
            pass
        del TrainingData[0]
        return TrainingData

    def GetTestingData_y(self):
        TestingData = {}
        try:
            with open('./dataset/test.tsv', 'r', newline='') as testing:
                reader = csv.reader(testing, delimiter='\t')
                for row in reader:
                    TestingData.update({f'{row[1]}': row[2]})
        except UnicodeDecodeError:
            pass
        del TestingData['path']
        return TestingData

    def GetTestingNames(self):
        TestingData = []
        try:
            with open('./dataset/test.tsv', 'r', newline='') as testing:
                reader = csv.reader(testing, delimiter='\t')
                for row in reader:
                    TestingData.append([f'./dataset/TestingSpectrograms/{row[1]}.png', row[2]])
        except UnicodeDecodeError:
            pass
        del TestingData[0]
        return TestingData

    def GenerateTrainingData_X(self):
        i = 0
        for i, item in enumerate(self.GetTrainingFileNames()):
            name = f'{item[0]}'
            ToSpectogram(mp3Path=f'./dataset/clips/{name}',
                         OutputImagePath=f'./dataset/TrainingSpectrograms/{name}.png')
            self.index += 1
            print(f'{i}: (Training) Converting {name} to Spectrogram')
        else:
            print(f'Training Convert Complete, {i} items')
            print(f'Total items processed: {self.index}')

    def GenerateTestingData_X(self):
        i = 0
        for i, item in enumerate(self.GetTestingNames()):
            name = item[0]
            ToSpectogram(mp3Path=f'./dataset/clips/{name}',
                         OutputImagePath=f'./dataset/TestingSpectrograms/{name}.png')
            i += 1
            print(f'{i}: (Testing) Converting {name} to Spectrogram')
        else:
            print(f'Testing Convert Complete, {i} items')
            print(f'Total items processed: {self.index}')


DataProcessor = ProcessData()
# DataProcessor.GenerateTestingData_X()
# DataProcessor.GenerateTestingData_X()
# Get Number of classes (files)1
NumClasses = len(os.listdir('./dataset/TrainingSpectrograms'))

BatchSize = 32
TargetImageSize = (332, 729)
# Generators
TrainingData = DataProcessor.GetTrainingFileNames()
DataTraining = pd.DataFrame(TrainingData, columns=['path','sentence'])
TestingData = DataProcessor.GetTestingNames()
DataTesting = pd.DataFrame(TestingData,columns=['path','sentence'])
DataGenerator = ImageDataGenerator(rescale=1/255)
UniqueClasses = DataTraining['sentence'].unique()
# Map each unique class to an integer label
ClassToLabel = {cls: label for label, cls in enumerate(UniqueClasses)}
# Map class labels to the 'sentence' column in DataTraining and DataTesting
DataTraining['label'] = DataTraining['sentence'].map(ClassToLabel)
DataTesting['label'] = DataTesting['sentence'].map(ClassToLabel)

# Train Generator
TrainGenerator = DataGenerator.flow_from_dataframe(
    dataframe=DataTraining,
    x_col='path',
    y_col='sentence',
    target_size=TargetImageSize,
    batch_size=BatchSize,
    class_mode='sparse',
    shuffle=True)
# Validation Generator
ValGenerator = DataGenerator.flow_from_dataframe(
    dataframe=DataTesting,
    x_col='path',
    y_col='sentence',
    target_size=TargetImageSize,
    batch_size=BatchSize,
    class_mode='sparse',
    shuffle=False)
# Create the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(332, 729, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Reshape((-1, 128)))
model.add(layers.LSTM(128, return_sequences=True))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(NumClasses, activation='softmax'))  # Change units to NumClasses
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
# Train the model
history = model.fit(TrainGenerator, epochs=200, validation_data=ValGenerator,)
# Save the model
model.save('./saved_model')
print("Model saved successfully.")
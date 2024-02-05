import os

import cv2
import keras.layers
import numpy as np
from os import listdir
import itertools
from matplotlib import pyplot as plt, units
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU
from sklearn.metrics import confusion_matrix
import matplotlib.font_manager as fm
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard
from sklearn.metrics import accuracy_score



# DATA_PATH = os.path.join('MP_Data')
# actions = np.array(['at', 'dworry', 'hi', 'hospital', 'school', 'thank', 'train_station', 'where'])
actions = np.array(
    ['happy', 'welcome', 'you', 'hao', 'at', 'school', 'mad', 'sad', 'sorry', 'dworry', 'train', 'hospital', 'what',
     'where', 'thanks', 'love'])
DATA_PATH = os.path.join('MP_Data')
MODEL_PATH = os.path.join('checkpoint')
FIGURE_PATH = os.path.join('figure')
words = {'at': '在', 'dworry': '沒關係', 'hospital': '醫', 'school': '學校',
         'thanks': '謝謝', 'train': '火車', 'where': '處', 'what': '什麼',
         'happy': '開心', 'welcome': '不客氣', 'you': '你', 'hao': '好', 'sorry': '對不起', 'mad': '生氣',
         'sad': '難過', 'love': '愛'}

label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []

# model_name = f'GRU_model1'
model_name = f'LSTM_model17'

translation_amount = 0.382
noise_std = 0.05
# tmp = ['what', 'thanks']
# for action in tmp:
#     action_path = os.path.join(DATA_PATH, action)
#     you_path = os.path.join(DATA_PATH, 'you')
#     for sequence in listdir(action_path):
#         sequence_path = os.path.join(action_path, sequence)
#         you_sequence_path = os.path.join(you_path, sequence)
#         for frame_num in listdir(sequence_path):
#             you_res = np.load(os.path.join(DATA_PATH, 'you', sequence, frame_num))
#             res = np.load(os.path.join(DATA_PATH, action, sequence, frame_num))
#
#             res[132:194] = you_res[132:194]
#             npy_path = os.path.join(action_path, str(sequence), str(frame_num))
#             np.save(npy_path, res)

for action in actions:  # data pre-processing
    action_path = os.path.join(DATA_PATH, action)
    print(action)
    for sequence in listdir(action_path):
        window = []
        sequence_path = os.path.join(action_path, sequence)
        for frame_num in listdir(sequence_path):
            res = np.load(os.path.join(DATA_PATH, action, sequence, frame_num))
            window.append(np.concatenate((res[40:88], res[132:])))

        sequences.append(window)
        labels.append(label_map[action])

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    print("Augment data : "+action)
    for sequence in listdir(action_path):
        window = []
        sequence_path = os.path.join(action_path, sequence)
        for frame_num in listdir(sequence_path):
            res = np.load(os.path.join(DATA_PATH, action, sequence, frame_num))

            res[40:88] += np.random.uniform(-translation_amount, translation_amount, size=48)
            res[132:] += np.random.uniform(-translation_amount, translation_amount, size=res[132:].shape)

            # res[40:88] += np.random.normal(0, noise_std, size=48)
            # res[132:] += np.random.normal(0, noise_std, size=res[132:].shape)

            window.append(np.concatenate((res[40:88], res[132:])))

        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)


model = Sequential()
if model_name == 'GRU_model1':
    model.add(GRU(64, return_sequences=True, activation='relu', input_shape=(30, 174)))
    model.add(GRU(128, return_sequences=True, activation='relu'))
    model.add(GRU(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
else:
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 174)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

tb_callback = TensorBoard(log_dir='Logs', histogram_freq=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-5)
checkpoint = ModelCheckpoint(filepath=f'{MODEL_PATH}/best_{model_name}.h5', monitor='val_loss', save_best_only=True)
callbacks_list = [tb_callback, reduce_lr, checkpoint]

history = model.fit(X_train, y_train, batch_size=32, epochs=30, callbacks=callbacks_list, validation_split=0.1)
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
model.summary()

plt.plot(train_loss, label='train_loss')
plt.plot(val_loss, label='val_loss')
plt.legend()
plt.title(f'{model_name} Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig(f'{FIGURE_PATH}/{model_name}_loss_curve.png')
plt.show()

plt.plot(train_acc, label='train_acc')
plt.plot(val_acc, label='val_acc')
plt.legend()
plt.title(f'{model_name} Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig(f'{FIGURE_PATH}/{model_name}_accuracy_curve.png')
plt.show()

font_path = 'font/NotoSansTC-Medium.otf'
prop = fm.FontProperties(fname=font_path)

y_pred = model.predict(X_test)
y_true_int = np.argmax(y_test, axis=1)
y_pred_int = np.argmax(y_pred, axis=1)

test_accuracy = accuracy_score(y_true_int, y_pred_int)
print("Test Accuracy = " + str(test_accuracy))

confusion_mat = confusion_matrix(y_true_int, y_pred_int)

plt.imshow(confusion_mat, cmap=plt.cm.Blues)
plt.title(f"{model_name} Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(actions))
plt.xticks(tick_marks, [words[action] for action in actions], rotation=45, fontproperties=prop)
plt.yticks(tick_marks, [words[action] for action in actions], fontproperties=prop)

thresh = confusion_mat.max() / 2.
for i, j in itertools.product(range(confusion_mat.shape[0]), range(confusion_mat.shape[1])):
    plt.text(j, i, format(confusion_mat[i, j], 'd'),
             horizontalalignment="center",
             color="white" if confusion_mat[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig(f'{FIGURE_PATH}/Diffusion_matrix_{model_name}.png')
plt.show()


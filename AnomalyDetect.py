mport pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from google.colab import drive
 
drive.mount('/content/drive')
 
normal_file_path = '/content/drive/My Drive/MYPATH/Normal Big Data.csv'
abnormal_file_path = '/content/drive/My Drive/MYPATH/Abnormal Big Data.csv'
 
normal_data = pd.read_csv(normal_file_path)
abnormal_data = pd.read_csv(abnormal_file_path)
 
def preprocess_data(data):
    data['TimeStamp'] = pd.to_datetime(data['TimeStamp'], format='%Y-%m-%d %H:%M:%S.%f')
    data['Hour'] = data['TimeStamp'].dt.hour
    data['Minute'] = data['TimeStamp'].dt.minute
    data['Second'] = data['TimeStamp'].dt.second
    label_encoder = LabelEncoder()
    data['ApplicationName_encoded'] = label_encoder.fit_transform(data['ApplicationName'])
    tfidf = TfidfVectorizer(max_features=100)
    message_tfidf = tfidf.fit_transform(data['Message']).toarray()
    numeric_data = data[['Hour', 'Minute', 'Second', 'ApplicationName_encoded']]
    numeric_data = np.hstack((numeric_data, message_tfidf))
    scaler = MinMaxScaler()
    numeric_data_scaled = scaler.fit_transform(numeric_data)
    return numeric_data_scaled
 
normal_data_scaled = preprocess_data(normal_data)
abnormal_data_scaled = preprocess_data(abnormal_data)
 
X_train, X_test = train_test_split(normal_data_scaled, test_size=0.2, random_state=42)
 
class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        activation_fn = 'relu'
        self.encoder = tf.keras.Sequential([
            layers.Dense(32, activation=activation_fn),
            layers.Dense(16, activation=activation_fn),
            layers.Dense(8, activation=activation_fn)
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(16, activation=activation_fn),
            layers.Dense(32, activation=activation_fn),
            layers.Dense(X_train.shape[1], activation='sigmoid')
        ])
 
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
 
def create_autoencoder_model():
    model = Autoencoder()
    opt = tf.keras.optimizers.RMSprop()
    loss = tf.keras.losses.Huber()
    model.compile(optimizer=opt, loss=loss)
    return model
 
epochs = 20
batch_size = 64
 
print(f"Training with activation: relu, optimizer: RMSprop, loss: Huber, epochs: {epochs}, batch size: {batch_size}")
model = create_autoencoder_model()
history = model.fit(
    X_train, X_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_test, X_test),
    verbose=1
)
 
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title(f'Model Loss (Activation: ReLU, Optimizer: RMSProp, Loss: Huber, Epochs: {epochs}, Batch Size: {batch_size})')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()
 
X_train_pred = model.predict(X_train)
reconstruction_loss_train = np.mean(np.abs(X_train - X_train_pred), axis=1)
 
X_test_pred = model.predict(X_test)
reconstruction_loss_test = np.mean(np.abs(X_test - X_test_pred), axis=1)
 
anomalous_data_pred = model.predict(abnormal_data_scaled)
reconstruction_loss_anomalous = np.mean(np.abs(abnormal_data_scaled - anomalous_data_pred), axis=1)
 
threshold = np.percentile(reconstruction_loss_train, 95)
 
anomalies_test = reconstruction_loss_test > threshold
anomalies_anomalous = reconstruction_loss_anomalous > threshold
 
plt.figure(figsize=(10, 6))
plt.hist(reconstruction_loss_train, bins=50, alpha=0.6, label='Normal Data (Train)')
plt.hist(reconstruction_loss_test, bins=50, alpha=0.6, label='Normal Data (Test)')
plt.hist(reconstruction_loss_anomalous, bins=50, alpha=0.6, label='Anomalous Data')
plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
plt.title('Reconstruction Error')
plt.xlabel('Error')
plt.ylabel('Number of Samples')
plt.legend()
plt.show()
 
print(f"Anomalies in test data: {sum(anomalies_test)} / {len(X_test)}")
print(f"Anomalies in anomalous data: {sum(anomalies_anomalous)} / {len(abnormal_data_scaled)}")

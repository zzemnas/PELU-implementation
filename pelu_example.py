import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras import losses
import pandas as pd
from sklearn import model_selection
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import matplotlib.pyplot as plt
import wget

# PELU activation function class
class PELU(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(PELU, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.alpha = self.add_weight(shape=(self.units,),
                               initializer=tf.keras.initializers.Constant(1),
                               trainable=True, dtype='float32' , constraint=tf.keras.constraints.MinMaxNorm(min_value=0.1, max_value=np.infty))
        
        self.beta = self.add_weight(shape=(self.units,),
                               initializer=tf.keras.initializers.Constant(0.5),
                               trainable=True, dtype='float32' , constraint=tf.keras.constraints.MinMaxNorm(min_value=0.1,max_value=np.infty))

    def call(self, inputs):
        first = (inputs)*(self.alpha/self.beta)
        second = self.alpha*(tf.math.exp(inputs/self.beta)-1)
        return tf.where(tf.math.greater_equal(inputs,0), first, second)

def grad_pelu_a(h,a,b):
  first = h/b
  second = tf.math.exp(h/b)-1
  return tf.where(tf.math.greater_equal(h,0), first, second)

def take_first_element(xb, yb):
  return xb

pelu = PELU(units=1)

x_range = tf.linspace(-5, 5, 200) # An equispaced grid of 200 points in [-5, +5]
y_range = pelu(x_range) #PELU function computed on the whole grid

plt.plot(x_range.numpy(), y_range.numpy())

# automatic differentiation precision check
with tf.GradientTape() as tape:
  f=pelu(x_range)

df_da = tape.jacobian(f,pelu.alpha)
x_range = tf.cast(x_range, dtype='float32')

exact_grad = tf.reshape(grad_pelu_a(x_range,a=pelu.alpha,b=pelu.beta),shape=[200,1])
tf.reduce_all(tf.abs(df_da - exact_grad)<1e-4)

model = tf.keras.Sequential(layers=[
      tf.keras.layers.Dense(50),
      PELU(50),
      tf.keras.layers.Dense(10, activation='softmax')
])

sgd = optimizers.SGD(learning_rate=1e-4, decay=1e-6, momentum=0.9, nesterov=True, clipvalue=0.1)
cross_entropy = losses.SparseCategoricalCrossentropy()
acc = metrics.SparseCategoricalAccuracy()

# ! wget https://archive.ics.uci.edu/ml/machine-learning-databases/00325/Sensorless_drive_diagnosis.txt

# pelu in practice
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00325/Sensorless_drive_diagnosis.txt'
filename = wget.download(url)
sensorless = pd.read_csv(filename, header=None, sep=' ')
X = sensorless.values[:, 0:-1].astype('float32')
y = sensorless.values[:, -1:].astype('int64') - 1

X_tr, X_tst, y_tr, y_tst = model_selection.train_test_split(X, y, stratify=y)
train_dataset = tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
train_dataset = train_dataset.shuffle(1000).batch(128)

normalizer = Normalization()
normalizer.adapt(train_dataset.map(take_first_element))

model = tf.keras.Sequential(layers=[
      normalizer,
      tf.keras.layers.Dense(50),
      PELU(50),
      tf.keras.layers.Dense(11,activation='softmax')
])

test_dataset = tf.data.Dataset.from_tensor_slices((X_tst, y_tst)).batch(128)
model.compile(optimizer=sgd, loss=cross_entropy, metrics=[acc])
my_fit = model.fit(train_dataset,epochs=20,validation_data=test_dataset)

plt.plot(my_fit.history['sparse_categorical_accuracy'])
plt.title('Model Accuracy With PELU as activation function.')
plt.xlabel('Epochs')

model_2 = tf.keras.Sequential(layers=[
      normalizer,
      tf.keras.layers.Dense(50,activation='relu'),
      tf.keras.layers.Dense(11,activation='softmax')
])

model_2.compile(optimizer=sgd, loss=cross_entropy, metrics=[acc])
my_fit_relu = model_2.fit(train_dataset,epochs=20,validation_data=test_dataset)
plt.plot(my_fit_relu.history['sparse_categorical_accuracy'])
plt.title('Model Accuracy With ReLU as activation function.')
plt.xlabel('Epochs')
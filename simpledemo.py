import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Dense, MaxPool2D,BatchNormalization,Conv2D,Reshape,Flatten,Dropout,Input,Activation
from tensorflow.keras import Model

# 自定义损失函数
def custom_mean_squared_error(y_true, y_pred):
	return tf.math.reduce_mean(tf.square(y_true - y_pred))

def custom_Bi_CategoricalCrossentropy(y_true, y_pred):
	t_loss = (-1)*(y_true * K.log(y_pred))
	return K.mean(t_loss)

# 神经网络模型定义
InPut = Input((100))
layer1 = Dense(10)(InPut)
Out = Dense(2)(layer1)

model = Model(inputs = InPut,outputs = Out)

# 神经网络模型参数设置
model.compile(optimizer=tf.keras.optimizers.Adam(0.005),
			loss=custom_Bi_CategoricalCrossentropy,
			metrics=['categorical_accuracy'])


model.summary()

# 后面用基于fit利用数据集训练model
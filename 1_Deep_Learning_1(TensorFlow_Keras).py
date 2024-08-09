# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 08:06:36 2024

@author: Naga Nitish
"""

import pandas as pd
df = pd.read_csv('1_insurance.csv')
df

# charges is our target variable
# sex and region are categorical variables


# 	age	sex	bmi	children	smoker	region	charges
# 0	19	female	27.900	0	yes	southwest	16884.92400
# 1	18	male	33.770	1	no	southeast	1725.55230
# 2	28	male	33.000	3	no	southeast	4449.46200
# 3	33	male	22.705	0	no	northwest	21984.47061
# 4	32	male	28.880	0	no	northwest	3866.85520
# ...	...	...	...	...	...	...	...
# 1333	50	male	30.970	3	no	northwest	10600.54830
# 1334	18	female	31.920	0	no	northeast	2205.98080
# 1335	18	female	36.850	0	no	southeast	1629.83350
# 1336	21	female	25.800	0	no	southwest	2007.94500
# 1337	61	female	29.070	0	yes	northwest	29141.36030
# 1338 rows × 7 columns

# =============================================================================

# Turn all categorical variables into numbers

df = pd.get_dummies(df)
df.head()

# 	age	bmi	children	charges	sex_female	sex_male	smoker_no	smoker_yes	region_northeast	region_northwest	region_southeast	region_southwest
# 0	19	27.900	0	16884.92400	True	False	False	True	False	False	False	True
# 1	18	33.770	1	1725.55230	False	True	True	False	False	False	True	False
# 2	28	33.000	3	4449.46200	False	True	True	False	False	False	True	False
# 3	33	22.705	0	21984.47061	False	True	True	False	False	True	False	False
# 4	32	28.880	0	3866.85520	False	True	True	False	False	True	False	False

# =============================================================================

# creating X and Y variables
y = df['charges']
x = df.drop('charges', axis=1)

# =============================================================================


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)
X_scaled

# array([[-1.43876426, -0.45332   , -0.90861367, ..., -0.56641788,
#         -0.61132367,  1.76548098],
#        [-1.50996545,  0.5096211 , -0.07876719, ..., -0.56641788,
#          1.63579466, -0.56641788],
#        [-0.79795355,  0.38330685,  1.58092576, ..., -0.56641788,
#          1.63579466, -0.56641788],
#        ...,
#        [-1.50996545,  1.0148781 , -0.90861367, ..., -0.56641788,
#          1.63579466, -0.56641788],
#        [-1.29636188, -0.79781341, -0.90861367, ..., -0.56641788,
#         -0.61132367,  1.76548098],
#        [ 1.55168573, -0.26138796, -0.90861367, ...,  1.76548098,
#         -0.61132367, -0.56641788]])

# =============================================================================

# Create Training and Testing sets

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# =============================================================================

import tensorflow as tf
import matplotlib.pyplot as plt 

# Create a new Model 
insurance_model = tf.keras.Sequential([
    tf.keras.layers.Dense(11),   # 11 x columns, 11 input layers
    tf.keras.layers.Dense(1)   # 1 y variable
])
# defines a simple neural network model using TensorFlow's Keras API
# tf.keras.Sequential:
#     This is a type of model in Keras that allows you to build a neural network by stacking layers 
# in a linear (sequential) order. Each layer in the list is added on top of the previous layer.
#     The Sequential model is easy to use for building feedforward neural networks where the layers 
# are arranged in a straight path from input to output.


# Compile the Model 
insurance_model.compile(loss=tf.keras.losses.mae,    # mean absolute error inplace of mean squared error
                        optimizer=tf.keras.optimizers.SGD(),  # Socastic Gradient Descent
                        metrics=['mae'])

# To regenerate the better weight values, we applying the procedure of identifying the best weights ---> 
# that method is Socastic Gradient Descent(SGD)

# =============================================================================

# Fit the Model 
insurance_model.fit(X_train,Y_train, epochs=100)

# Epoch 1/100
# 34/34 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 13083.9170 - mae: 13083.9170
# Epoch 2/100
# 34/34 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 13403.9902 - mae: 13403.9902
# Epoch 3/100
# 34/34 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 13410.0215 - mae: 13410.0215
# Epoch 4/100
# 34/34 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 13229.2871 - mae: 13229.2871
# Epoch 5/100
# 34/34 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 13003.1885 - mae: 13003.1885
# Epoch 6/100
# 34/34 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 13941.8633 - mae: 13941.8633 
# Epoch 7/100
# 34/34 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 13830.5703 - mae: 13830.5703 
# Epoch 8/100
# ............................................................................
# Epoch 96/100
# 34/34 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 3286.4775 - mae: 3286.4775 
# Epoch 97/100
# 34/34 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 3485.2036 - mae: 3485.2036 
# Epoch 98/100
# 34/34 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 3685.8508 - mae: 3685.8508
# Epoch 99/100
# 34/34 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 3400.3162 - mae: 3400.3162
# Epoch 100/100
# 34/34 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 3387.7471 - mae: 3387.7471
# <keras.src.callbacks.history.History at 0x7cb3b1ea2aa0>

# =============================================================================


mae_values = insurance_model.history.history['mae']
mae_values

# [13345.7021484375,
#  13344.7568359375,
#  13343.419921875,
#  13341.2158203125,
#  13337.2578125,
#  13329.8232421875,
#  13315.52734375,
#  13287.771484375,
#  13233.419921875,
#  13126.8974609375,
#  12917.6630859375,
#  12505.94140625,
#  11726.890625,
#  10525.8955078125,
#  .................]

# =============================================================================

# See the error how it is reduced to minimum times for epoch = 10
import matplotlib.pyplot as plt
plt.figure(figsize=(15,4))
plt.scatter(range(1,101), mae_values)
plt.plot(range(1,101), mae_values,color='red')
plt.ylabel('MAE')
plt.xlabel('Epochs')
# plt.show()

# =============================================================================

# Set Random Seed
tf.random.set_seed(42)
# The statement tf.random.set_seed(42) in TensorFlow sets the global random seed to the value 42. 
# This ensures that the results of any random operations (like initializing weights, shuffling data, 
# or generating random numbers) are reproducible.

# By setting a seed, you ensure that every time you run your code, the random operations produce the 
# same results, which is important for debugging and sharing your experiments with others.  

# =============================================================================

# Now here we are adding hidden layer and increase number of units 
insurance_model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(11),
    tf.keras.layers.Dense(15),   # Hidden layers should be more than input layers
    tf.keras.layers.Dense(1)
])

# Compile the Model
insurance_model_2.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.SGD(),
                        metrics=['mae'])

# =============================================================================

# Fit the Model
insurance_model_2.fit(X_train,Y_train, epochs=200)

# Epoch 1/200
# 34/34 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 13298.2900 - mae: 13298.2900
# Epoch 2/200
# 34/34 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 13296.5293 - mae: 13296.5293
# Epoch 3/200
# 34/34 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 13289.2441 - mae: 13289.2441 
# Epoch 4/200
# 34/34 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 12703.3115 - mae: 12703.3115 
# Epoch 5/200
# 34/34 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 4755.7183 - mae: 4755.7183
# Epoch 6/200
# 34/34 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 4697.2612 - mae: 4697.2612
# ................................................................

# =============================================================================

mae_values = insurance_model_2.history.history['mae']
len(mae_values)
# 200

# =============================================================================

# See the error how it is reduced to miniumum 

import matplotlib.pyplot as plt 
plt.figure(figsize=(15,4))
plt.scatter(range(1,201), mae_values)
plt.plot(range(1,201), mae_values,color='red')
plt.ylabel('MAE')
plt.xlabel('Epochs')
plt.show()

# =============================================================================

# ALSO AFTER ADDING HIDDEN LAYER ALSO ERROR IS NOT REDUCING
# MODEL IS FITTED TO THE IRREDUCIBLE ERROR 

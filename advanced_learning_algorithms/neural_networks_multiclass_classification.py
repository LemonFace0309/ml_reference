# UNQ_C1
# GRADED CELL: my_softmax

def my_softmax(z):  
    """ Softmax converts a vector of values to a probability distribution.
    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """    
    ### START CODE HERE ### 

    ez = np.exp(z)
    sum = np.sum(ez)
    a = ez / sum
    
    ### END CODE HERE ### 
    return a

# UNQ_C2
# GRADED CELL: Sequential model

tf.random.set_seed(1234) # for consistent results
model = Sequential(
    [               
        ### START CODE HERE ###
        tf.keras.Input(shape=(400,)),
        Dense(25, activation="relu", name="layer1"),
        Dense(15, activation="relu", name="layer2"),
        Dense(10, activation="linear", name="layer3"),
        ### END CODE HERE ### 
    ], name = "my_model" 
)


# model.compile(
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
# )

# history = model.fit(
#     X,y,
#     epochs=40
# )

# image_of_two = X[1015]

# prediction = model.predict(image_of_two.reshape(1,400))  # prediction

# print(f" predicting a Two: \n{prediction}")
# print(f" Largest Prediction index: {np.argmax(prediction)}")

# prediction_p = tf.nn.softmax(prediction)

# print(f" predicting a Two. Probability vector: \n{prediction_p}")
# print(f"Total of predictions: {np.sum(prediction_p):0.3f}")


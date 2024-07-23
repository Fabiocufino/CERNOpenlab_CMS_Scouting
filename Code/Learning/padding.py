import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Masking, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Example sequences of different lengths
sequences = [
    [1, 2, 3],
    [4, 5, 6, 7],
    [8, 9]
]

# Pad sequences to the same length (e.g., length 4)
padded_sequences = pad_sequences(sequences, padding='post', maxlen=4, value=0)
print("Padded Sequences:\n", padded_sequences)

# Example labels
labels = np.array([0, 1, 0])

# Define the input shape
input_shape = (4,)  # After padding, all sequences have length 4

# Create input layer
input_layer = Input(shape=input_shape, name='input_layer')

# Create a masking layer to ignore padded values (assume 0 is the padding value)
masking_layer = Masking(mask_value=0)(input_layer)

print("Masked Sequences:\n", masking_layer)
# Add a simple dense layer
dense_layer = Dense(10, activation='relu')(masking_layer)

# Flatten the output
flatten_layer = Flatten()(dense_layer)

# Final output layer
output_layer = Dense(1, activation='sigmoid')(flatten_layer)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
model.fit(padded_sequences, labels, epochs=10, batch_size=1)

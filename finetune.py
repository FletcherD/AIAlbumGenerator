"""
Datafile is a text file with one sentence per line _DATASETS/data.txt
tf_gpt2_keras_lora is the name of the fine-tuned model
"""

import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from transformers.modeling_tf_utils import get_initializer
import os

# use 2 cores
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# Use pretrained model if it exists
# otherwise download it
if os.path.exists("tf_gpt2_keras_lora"):
    print("Model exists")
    # use pretrained model
    model = TFGPT2LMHeadModel.from_pretrained("tf_gpt2_keras_lora")
else:
    print("Downloading model")
    model = TFGPT2LMHeadModel.from_pretrained("gpt2-large")

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")

# Load and preprocess the data
with open("dataset.txt", "r") as f:
    lines = f.read().split("\n")

# Encode the data using the tokenizer and truncate the sequences to a maximum length of 1024 tokens
input_ids_all = []
for line in lines:
    encoding = tokenizer.encode(line, add_special_tokens=True, max_length=1024, truncation=True)
    input_ids_all.append(encoding)

# Define some params
batch_size = 2
num_epochs = 3
learning_rate = 5e-5

# Define the optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Fine-tune the model using low-rank adaptation and attention pruning
for layer in model.transformer.h:
    layer.attention_output_dense = tf.keras.layers.Dense(units=256, kernel_initializer=get_initializer(0.02), name="attention_output_dense")
    
model.summary()

# Train the model
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    
    # Shuffle the input data
    input_ids_all = tf.random.shuffle(input_ids_all)
    input_ids = input_ids_all[:65536]
    
    for i in range(0, len(input_ids), batch_size):
        batch = input_ids[i:i+batch_size]
        # Pad the batch to the same length
        batch = tf.keras.preprocessing.sequence.pad_sequences(batch, padding="post")
        # Define the inputs and targets
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        # Compute the predictions and loss
        with tf.GradientTape() as tape:
            logits = model(inputs)[0]
            loss = loss_fn(targets, logits)
        # Compute the gradients and update the parameters
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Print the loss every 10 batches
        if i % (10 * batch_size) == 0:
            print(f"Batch {i}/{len(input_ids)} - loss: {loss:.4f}")
            
# Save the fine-tuned model
model.save_pretrained("tf_gpt2_keras_lora")

# Generate text using the fine-tuned model
input_ids = tokenizer.encode("How much wood", return_tensors="tf")
output = model.generate([], max_length=100, do_sample=True, top_k=50, top_p=0.95, temperature=0.9)
print(tokenizer.decode(output[0], skip_special_tokens=True))

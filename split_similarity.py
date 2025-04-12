import pickle
import numpy as np
import os

# Load the similarity matrix
with open('similarity.pkl', 'rb') as f:
    similarity = pickle.load(f)

# Get the shape of the matrix
shape = similarity.shape
print(f"Similarity matrix shape: {shape}")

# Split the matrix into 4 parts
half_row = shape[0] // 2
half_col = shape[1] // 2

part1 = similarity[:half_row, :half_col]
part2 = similarity[:half_row, half_col:]
part3 = similarity[half_row:, :half_col]
part4 = similarity[half_row:, half_col:]

# Save each part
with open('similarity_part1.pkl', 'wb') as f:
    pickle.dump(part1, f)
with open('similarity_part2.pkl', 'wb') as f:
    pickle.dump(part2, f)
with open('similarity_part3.pkl', 'wb') as f:
    pickle.dump(part3, f)
with open('similarity_part4.pkl', 'wb') as f:
    pickle.dump(part4, f)

print("Split complete. Now modify your app.py to load these parts.")
import struct

sentences = [
    "This is the first sentence.",
    "Here comes the second sentence.",
    "And this is the third sentence."
]

# Convert sentences to a single string
sentences_str = '\n'.join(sentences)

# Convert string to bytes using UTF-8 encoding
sentences_bytes = sentences_str.encode('utf-8')

# Get the length of the bytes
length = len(sentences_bytes)

# Pack the length and the bytes using struct.pack
packed_data = struct.pack('<I', length) + sentences_bytes

# Save packed data to .bin file
with open("sentences.bin", "wb") as file:
    file.write(packed_data)

import struct

text: str = "What we are doing is very irresponsible."
text_b: bytes = text.encode("UTF-8")
print(struct.pack("<I", len(text_b))+text_b)
# <I means little-endian unsigned integers, followed by the number of elements

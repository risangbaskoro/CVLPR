IMAGE_SHAPE = [94, 24]
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
CHARS_DICT = {idx: char for idx, char in enumerate(CHARS)}
DECODE_DICT = {char: idx for idx, char in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1

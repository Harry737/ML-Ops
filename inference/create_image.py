from PIL import Image, ImageDraw
import numpy as np

# Create black background
img = Image.new("L", (28, 28), color=0)
draw = ImageDraw.Draw(img)

# Draw thick digit
draw.text((6, 2), "9", fill=255)

# Convert to numpy
arr = np.array(img)

# Center digit using bounding box
coords = np.column_stack(np.where(arr > 0))
y0, x0 = coords.min(axis=0)
y1, x1 = coords.max(axis=0)

digit = arr[y0:y1+1, x0:x1+1]
digit = Image.fromarray(digit).resize((20, 20))

final = Image.new("L", (28, 28), color=0)
final.paste(digit, (4, 4))

final.save("digit_mnist_like.png")

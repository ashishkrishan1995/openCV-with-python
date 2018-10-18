import pytesseract
from PIL import Image 

img = Image.open('Untitled.png')
result = pytesseract.image_to_string(img)
print(result)

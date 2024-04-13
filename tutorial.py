#import pillow
from PIL import Image, ImageOps

# Open an image file
im = Image.open("image_processing/flower1.jpg")

# Display image
im.show()

#resize image 
im.thumbnail((128, 128))
im.show()

#relative resizing
size = (128, 128)

with Image.open("image_processing/flower1.jpg") as im:
    ImageOps.fit(im, size, Image.ANTIALIAS).show()
    ImageOps.cover(im, size).show()
    ImageOps.contain(im, size).show()
    ImageOps.pad(im, size).show()

#seperate bands 
r, g, b = im.split()
r.show()
g.show()
b.show()

#merge bands
im = Image.merge("RGB", (r, g, b))
im.show()


#convert to grayscale
im = ImageOps.grayscale(im)
im.show()

#crop image
im = Image.open("image_processing/flower1.jpg")
box = (100, 100, 400, 400)
im = im.crop(box)
im.show()

#point operations
im = Image.open("image_processing/flower1.jpg")

# apply point transformations
out = im.point(lambda i: int(i * 1.2))
out.show()

#compare transformation with original
out = Image.new("RGB", (512, 512))
out.paste(im, (0, 0))
out.paste(out, (256, 0))
out.show()


#apply another point transformation
out = out.point(lambda i: int(i + 30))
out.show()





from PIL import Image, ImageDraw

for i in range(10):
    # Load the image from the provided path
    img_path = f'city/city_{i}.jpg'
    img = Image.open(img_path)

    # Resize the image to 512x512
    img_resized = img.resize((512, 512), Image.LANCZOS)

    # Create a mask for the circle
    mask = Image.new('L', (512, 512), 0)
    draw = ImageDraw.Draw(mask) 
    draw.ellipse((0, 0, 512, 512), fill=255)

    # Apply the mask to the resized image to get the circle
    img_circle = Image.new("RGB", (512, 512), "#808080")
    img_circle.paste(img_resized, (0, 0), mask)

    # Save the result to a new file
    img_circle.save(img_path)

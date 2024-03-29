from PIL import Image, ImageDraw, ImageOps

def resize_and_crop_circle(image_path):
    # Load the original image
    original_image = Image.open(image_path)
    
    # Calculate the new width to maintain a 16:9 aspect ratio with height = 512
    new_width = int((512 / 9) * 16)
    
    # Resize the original image to the new dimensions
    resized_image = original_image.resize((new_width, 512), Image.Resampling.LANCZOS)
    
    # Calculate the coordinates for the square to be cropped from the center
    left = (resized_image.width - 512) / 2
    top = (resized_image.height - 512) / 2
    right = (resized_image.width + 512) / 2
    bottom = (resized_image.height + 512) / 2
    crop_box = (left, top, right, bottom)
    
    # Crop the 512x512 square from the center of the resized image
    cropped_square = resized_image.crop(crop_box)
    
    # Create a circular mask
    mask = Image.new('L', (512, 512), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, 512, 512), fill=255)
    
    # Create an image for the circle with a transparent background
    circle = Image.new('RGBA', (512, 512), (0,0,0,0))
    circle.paste(cropped_square, (0, 0), mask)
    
    # Create a new image with the same dimensions as the resized image, filled with #808080
    final_image = Image.new('RGB', (new_width, 512), '#808080')
    
    # Calculate the position to paste the circle onto the new image
    paste_x = (final_image.width - circle.width) // 2
    paste_y = (final_image.height - circle.height) // 2
    
    # Convert the circle image to RGB before pasting
    circle_rgb = Image.new("RGB", circle.size, (0, 0, 0))
    circle_rgb.paste(circle, mask=circle.split()[3])  # 3 is the alpha channel
    
    # Paste the circle onto the new image
    final_image.paste(circle_rgb, (paste_x, paste_y), mask)
    
    return final_image


for i in range(10):
    # Load the image from the provided path
    mnt_path = f'mountains/mountain_{i}.jpg'
    city_path = f'city/city_{i}.jpg'
    city_img = resize_and_crop_circle(city_path)
    mnt_img = resize_and_crop_circle(mnt_path)
    mnt_img.save(mnt_path)
    city_img.save(city_path)
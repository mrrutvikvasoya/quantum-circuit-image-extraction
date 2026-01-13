# Script to randomly select images from the current directory, convert some to black & white, and copy others as originals to a new folder.
import os
import random
import shutil
from PIL import Image


# Define source and destination directories
SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
DEST_DIR = os.path.join(SOURCE_DIR, 'newimages')

# Number of images to process
NUM_TOTAL = 550  # Total images to select
NUM_BW = 350     # Number of images to convert to black & white
NUM_ORIG = 200   # Number of images to copy as original


# Create destination directory if it doesn't exist
os.makedirs(DEST_DIR, exist_ok=True)


# Gather all image files in the source directory
IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
all_images = [f for f in os.listdir(SOURCE_DIR) if os.path.splitext(f)[1].lower() in IMAGE_EXTS]


# Check if there are enough images to process
if len(all_images) < NUM_TOTAL:
	print(f'Not enough images in the folder. Found {len(all_images)}, need {NUM_TOTAL}.')
	exit(1)



# Randomly select images and split into BW and original groups
selected_images = random.sample(all_images, NUM_TOTAL)
random.shuffle(selected_images)  # Shuffle to randomize order
bw_images = selected_images[:NUM_BW]
orig_images = selected_images[NUM_BW:NUM_BW+NUM_ORIG]



# Convert selected images to black & white and save to destination
for img_name in bw_images:
	src_path = os.path.join(SOURCE_DIR, img_name)
	dest_path = os.path.join(DEST_DIR, img_name)
	try:
		with Image.open(src_path) as img:
			bw_img = img.convert('L')
			bw_img.save(dest_path)
		print(f'BW processed and saved: {dest_path}')
	except Exception as e:
		print(f'Error processing {img_name}: {e}')

# Copy the remaining selected images as originals to destination
for img_name in orig_images:
	src_path = os.path.join(SOURCE_DIR, img_name)
	dest_path = os.path.join(DEST_DIR, img_name)
	try:
		import shutil
		shutil.copy2(src_path, dest_path)
		print(f'Original copied: {dest_path}')
	except Exception as e:
		print(f'Error copying {img_name}: {e}')

import os
import shutil
import cv2

# Paths
image_folder = "target_images"  # Update this
output_folder = "manual_segmentation_output"  # Update this
log_file = "processed_images.txt"

# Create rating folders if they don't exist
for i in range(1, 6):
    os.makedirs(os.path.join(output_folder, str(i)), exist_ok=True)

# Load processed images from the log file
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        processed_images = set(f.read().splitlines())
else:
    processed_images = set()

# Get list of images
images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
images.sort()  # Optional: Sort images by name

# Process images that are not in the log
for image in images:
    if image in processed_images:
        continue  # Skip already rated images

    image_path = os.path.join(image_folder, image)
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error loading image: {image}")
        continue

    cv2.imshow("Rate the Image (Press 1-5)", img)

    while True:
        key = cv2.waitKey(0) & 0xFF  # Wait for key press

        if key in [ord(str(i)) for i in range(1, 6)]:  # If 1-5 is pressed
            rating = chr(key)  # Convert key to string
            shutil.copy(image_path, os.path.join(output_folder, rating, image))  # Copy instead of move

            # Log the processed image
            with open(log_file, "a") as f:
                f.write(image + "\n")

            print(f"Copied {image} to folder {rating}")
            break  # Move to the next image
        elif key == 27:  # ESC key to exit
            print("Exiting...")
            cv2.destroyAllWindows()
            exit()

    cv2.destroyAllWindows()

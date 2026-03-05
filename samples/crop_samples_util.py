import os
from PIL import Image

def crop_samples_to_2x2(folder="samples"):
    """
    Finds specific sample grids in the samples folder and crops them 
    down to the top-left 4 images (a 2x2 grid) for the report.
    """
    if not os.path.exists(folder):
        print(f"Folder '{folder}' does not exist.")
        return

    # Count how many we process
    processed_count = 0

    for filename in os.listdir(folder):
        # Only target files ending in 'samples.png' or 'samples_05.png'
        is_target = filename.endswith("samples.png") or filename.endswith("samples_05.png")
        # Exclude PCA plots and already cropped files
        is_excluded = filename.endswith("pca.png") or filename.endswith("_cropped.png")

        if is_target and not is_excluded:
            filepath = os.path.join(folder, filename)
            try:
                img = Image.open(filepath)
                
                # MATH FOR THE CROP BOX:
                # Padding (2px) + Img (28px) + Padding (2px) + Img (28px) + Padding (2px) = 62px
                # Box format: (left, upper, right, lower)
                crop_box = (0, 0, 62, 62)
                
                # Crop and save
                cropped_img = img.crop(crop_box)
                new_filename = filename.replace(".png", "_cropped.png")
                new_filepath = os.path.join(folder, new_filename)
                
                cropped_img.save(new_filepath)
                print(f"✅ Cropped: {filename} -> {new_filename}")
                processed_count += 1
                
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

    if processed_count == 0:
        print("No matching images found to crop.")
    else:
        print(f"\nSuccessfully cropped {processed_count} images for your report!")

if __name__ == "__main__":
    crop_samples_to_2x2()
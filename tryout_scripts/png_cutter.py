from PIL import Image, ImageChops
import os


def crop_image_with_padding(image_folder_path, output_path, padding=20):
    for image_name in os.listdir(image_folder_path):
        if image_name.endswith(".png"):
            image_path = os.path.join(image_folder_path, image_name)
    
        img = Image.open(image_path).convert("RGB")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        diff = ImageChops.difference(img, bg)

        bbox = diff.getbbox()

        if bbox:
            pad = padding  # Pixel Rand
            left, upper, right, lower = bbox
            left = max(0, left - pad)
            upper = max(0, upper - pad)
            right = min(img.width, right + pad)
            lower = min(img.height, lower + pad)

        cropped = img.crop((left, upper, right, lower))
        cropped.save(os.path.join(output_path, f"{image_name}"))
        
if __name__ == "__main__":
    image_folder_path = "outputs/figs_NiO_scaled"
    output_path = "outputs/figs_NiO_scaled_cropped"
    # image_folder_path = "outputs/figs_NiO_not_scaled"
    # output_path = "outputs/figs_NiO_not_scaled_cropped"
    os.makedirs(output_path, exist_ok=True)
    crop_image_with_padding(image_folder_path, output_path)
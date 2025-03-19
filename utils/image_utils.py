import os
import numpy as np
from PIL import Image, ImageOps

def load_image(image_path):
    """
    Load an image from a path.
    
    Args:
        image_path (str): Path to the image
        
    Returns:
        PIL.Image: Loaded image
    """
    try:
        img = Image.open(image_path)
        # Convert to RGBA to handle transparency
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        raise

def save_image(image, output_path):
    """
    Save an image to a path.
    
    Args:
        image (PIL.Image): Image to save
        output_path (str): Path to save the image
        
    Returns:
        str: Path to the saved image
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save image
        image.save(output_path)
        
        return output_path
    except Exception as e:
        print(f"Error saving image to {output_path}: {str(e)}")
        raise

def resize_image(image, size):
    """
    Resize an image to a specific size.
    
    Args:
        image (PIL.Image): Image to resize
        size (tuple): Target size (width, height)
        
    Returns:
        PIL.Image: Resized image
    """
    try:
        # Resize the image
        return image.resize(size, Image.LANCZOS)
    except Exception as e:
        print(f"Error resizing image: {str(e)}")
        raise

def create_transparent_overlay(base_image, overlay_image, opacity=0.7):
    """
    Create a transparent overlay of one image on another.
    
    Args:
        base_image (PIL.Image): Base image
        overlay_image (PIL.Image): Overlay image
        opacity (float): Opacity of the overlay (0.0-1.0)
        
    Returns:
        PIL.Image: Combined image
    """
    try:
        # Ensure images are the same size
        if base_image.size != overlay_image.size:
            overlay_image = resize_image(overlay_image, base_image.size)
        
        # Create a copy of the base image
        result = base_image.copy()
        
        # Create a new image with the same size but with an alpha channel
        overlay = Image.new('RGBA', base_image.size, (0, 0, 0, 0))
        
        # Paste the overlay onto the new image
        overlay.paste(overlay_image, (0, 0), overlay_image)
        
        # Apply opacity to the overlay
        overlay_array = np.array(overlay)
        overlay_array[:, :, 3] = (overlay_array[:, :, 3] * opacity).astype(np.uint8)
        overlay = Image.fromarray(overlay_array)
        
        # Composite the overlay onto the base image
        result = Image.alpha_composite(result, overlay)
        
        return result
    except Exception as e:
        print(f"Error creating transparent overlay: {str(e)}")
        raise

def extract_foreground(image, threshold=240):
    """
    Extract the foreground from an image by removing white/light background.
    
    Args:
        image (PIL.Image): Image to process
        threshold (int): Threshold for considering a pixel as background (0-255)
        
    Returns:
        PIL.Image: Image with transparent background
    """
    try:
        # Convert to RGBA if not already
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Get image data as numpy array
        data = np.array(image)
        
        # Create alpha mask: make white/light pixels transparent
        r, g, b, a = data.T
        white_areas = (r > threshold) & (g > threshold) & (b > threshold)
        data[..., 3][white_areas.T] = 0
        
        # Create new image with the modified data
        return Image.fromarray(data)
    except Exception as e:
        print(f"Error extracting foreground: {str(e)}")
        raise

def crop_to_content(image, padding=10):
    """
    Crop an image to remove unnecessary background and focus on the content.
    
    Args:
        image (PIL.Image): Image to crop
        padding (int): Padding to add around the content
        
    Returns:
        PIL.Image: Cropped image
    """
    try:
        # Convert to RGBA if not already
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Get the alpha channel
        alpha = image.split()[3]
        
        # Get the bounding box of the non-zero regions
        bbox = ImageOps.invert(alpha).getbbox()
        
        if bbox:
            # Add padding
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.width, x2 + padding)
            y2 = min(image.height, y2 + padding)
            
            # Crop the image
            return image.crop((x1, y1, x2, y2))
        else:
            return image
    except Exception as e:
        print(f"Error cropping to content: {str(e)}")
        raise

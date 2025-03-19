import os
import sys
import requests
import tempfile
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torchvision.transforms as transforms
from dotenv import load_dotenv

class VirtualTryOn:
    """
    Interface for GP-VTON (Geometry Parsing Virtual Try-On Network) model.
    
    GP-VTON is a state-of-the-art virtual try-on model that accurately handles
    different body types and clothing sizes while preserving high visual fidelity.
    """
    
    def __init__(self):
        """Initialize the virtual try-on model."""
        # Load environment variables
        load_dotenv()
        
        # Check for API key
        self.api_key = os.getenv("HUGGINGFACE_API_TOKEN")
        if not self.api_key:
            print("Warning: HUGGINGFACE_API_TOKEN not found in environment variables.")
        
        # Model parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_initialized = False
        self.model = None
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def _initialize_model(self):
        """
        Initialize the GP-VTON model.
        
        This is done lazily to avoid loading the model if it's not used.
        """
        try:
            # In a real implementation, you would import and load the GP-VTON model here
            # For example:
            # from gpvton.models import GPVTONModel
            # self.model = GPVTONModel()
            # self.model.load_state_dict(torch.load("gpvton_model.pth"))
            # self.model.to(self.device)
            # self.model.eval()
            
            # For the purpose of this code, we'll simulate model initialization
            print("Initializing GP-VTON model...")
            self.model_initialized = True
            
        except Exception as e:
            print(f"Error initializing GP-VTON model: {str(e)}")
            self.model_initialized = False
    
    def _preprocess_images(self, person_image_path, clothing_image_path):
        """
        Preprocess images for the GP-VTON model.
        
        Args:
            person_image_path (str): Path to the person image
            clothing_image_path (str): Path to the clothing image
            
        Returns:
            tuple: Preprocessed person and clothing tensors
        """
        # Load images
        person_img = Image.open(person_image_path).convert('RGB')
        clothing_img = Image.open(clothing_image_path).convert('RGB')
        
        # Resize images to model input size (typically 256x192 for most try-on models)
        person_img = person_img.resize((192, 256))
        clothing_img = clothing_img.resize((192, 256))
        
        # Apply transformations
        person_tensor = self.transform(person_img).unsqueeze(0).to(self.device)
        clothing_tensor = self.transform(clothing_img).unsqueeze(0).to(self.device)
        
        return person_tensor, clothing_tensor
    
    def _generate_warped_cloth(self, person_tensor, clothing_tensor, fit_analysis):
        """
        Generate a warped clothing image based on the person's body and fit analysis.
        
        Args:
            person_tensor (torch.Tensor): Preprocessed person tensor
            clothing_tensor (torch.Tensor): Preprocessed clothing tensor
            fit_analysis (dict): Analysis of the fit
            
        Returns:
            torch.Tensor: Warped clothing tensor
        """
        # In a real implementation, you would use the GP-VTON model here
        # For example:
        # with torch.no_grad():
        #     warped_cloth = self.model.warp_module(person_tensor, clothing_tensor)
        
        # For the purpose of this code, we'll simulate warping based on fit analysis
        overall_fit = fit_analysis.get("overall_fit", "good")
        
        # Simulate different warping based on fit
        if "loose" in overall_fit:
            # Scale factor for loose fit (clothing appears larger)
            scale_factor = 1.1
        elif "tight" in overall_fit:
            # Scale factor for tight fit (clothing appears smaller/tighter)
            scale_factor = 0.9
        else:
            # Default scale factor for good fit
            scale_factor = 1.0
        
        # For simulation purposes only - in a real implementation, this would be handled
        # by the GP-VTON model's warping module
        return clothing_tensor  # In reality, this would be the warped clothing
    
    def _apply_try_on(self, person_tensor, warped_cloth_tensor):
        """
        Apply the try-on process to generate the final image.
        
        Args:
            person_tensor (torch.Tensor): Preprocessed person tensor
            warped_cloth_tensor (torch.Tensor): Warped clothing tensor
            
        Returns:
            torch.Tensor: Generated try-on image tensor
        """
        # In a real implementation, you would use the GP-VTON model here
        # For example:
        # with torch.no_grad():
        #     try_on_img = self.model.try_on_module(person_tensor, warped_cloth_tensor)
        
        # For the purpose of this code, we'll simulate try-on
        # In reality, the GP-VTON model would handle this with its sophisticated algorithms
        return person_tensor  # In reality, this would be the try-on result
    
    def _postprocess_image(self, tensor):
        """
        Convert a tensor to a PIL Image.
        
        Args:
            tensor (torch.Tensor): Image tensor
            
        Returns:
            PIL.Image: Converted image
        """
        # Denormalize
        tensor = tensor.squeeze(0).cpu().clone()
        tensor = tensor * 0.5 + 0.5
        tensor = tensor.clamp(0, 1)
        
        # Convert to PIL Image
        array = tensor.permute(1, 2, 0).numpy() * 255
        return Image.fromarray(array.astype(np.uint8))
    
    def generate_tryon_image(self, person_image_path, clothing_image_path, fit_analysis, output_path=None):
        """
        Generate a virtual try-on image of a person wearing a clothing item.
        
        Args:
            person_image_path (str): Path to the person image
            clothing_image_path (str): Path to the clothing image
            fit_analysis (dict): Analysis of the fit
            output_path (str, optional): Path to save the output image
            
        Returns:
            str: Path to the generated image
        """
        try:
            # Initialize model if not already initialized
            if not self.model_initialized:
                self._initialize_model()
            
            # If we can't initialize the model, use the HuggingFace API instead
            if not self.model_initialized:
                return self._generate_using_api(person_image_path, clothing_image_path, fit_analysis, output_path)
            
            # Preprocess images
            person_tensor, clothing_tensor = self._preprocess_images(person_image_path, clothing_image_path)
            
            # Generate warped clothing
            warped_cloth_tensor = self._generate_warped_cloth(person_tensor, clothing_tensor, fit_analysis)
            
            # Apply try-on
            result_tensor = self._apply_try_on(person_tensor, warped_cloth_tensor)
            
            # Postprocess result
            result_img = self._postprocess_image(result_tensor)
            
            # Add fit annotation
            result_img = self._add_fit_annotation(result_img, fit_analysis)
            
            # Save the result
            if output_path is None:
                output_path = os.path.join(tempfile.gettempdir(), "tryon_result.png")
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result_img.save(output_path)
            
            return output_path
            
        except Exception as e:
            print(f"Error generating try-on image: {str(e)}")
            # Fall back to using the API
            return self._generate_using_api(person_image_path, clothing_image_path, fit_analysis, output_path)
    
    def _generate_using_api(self, person_image_path, clothing_image_path, fit_analysis, output_path=None):
        """
        Generate a try-on image using the HuggingFace API.
        
        Args:
            person_image_path (str): Path to the person image
            clothing_image_path (str): Path to the clothing image
            fit_analysis (dict): Analysis of the fit
            output_path (str, optional): Path to save the output image
            
        Returns:
            str: Path to the generated image
        """
        try:
            # Check if API key is available
            if not self.api_key:
                raise ValueError("HUGGINGFACE_API_TOKEN not set in environment variables")
            
            # HuggingFace API endpoint for GP-VTON
            # Note: This is a placeholder URL, you would need to use the actual endpoint
            API_URL = "https://api-inference.huggingface.co/models/user/gp-vton"
            
            # Set up headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Prepare data
            with open(person_image_path, "rb") as f:
                person_bytes = f.read()
            
            with open(clothing_image_path, "rb") as f:
                clothing_bytes = f.read()
            
            # Send request
            data = {
                "person_image": person_bytes,
                "clothing_image": clothing_bytes,
                "fit_type": fit_analysis.get("overall_fit", "good")
            }
            
            response = requests.post(API_URL, headers=headers, json=data)
            
            # Check response
            if response.status_code == 200:
                # Save response image
                if output_path is None:
                    output_path = os.path.join(tempfile.gettempdir(), "tryon_result.png")
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                with open(output_path, "wb") as f:
                    f.write(response.content)
                
                # Add fit annotation
                img = Image.open(output_path)
                img = self._add_fit_annotation(img, fit_analysis)
                img.save(output_path)
                
                return output_path
            else:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")
            
        except Exception as e:
            print(f"Error generating try-on image with API: {str(e)}")
            
            # Final fallback: use simple image composition
            return self._generate_fallback(person_image_path, clothing_image_path, fit_analysis, output_path)
    
    def _generate_fallback(self, person_image_path, clothing_image_path, fit_analysis, output_path=None):
        """
        Generate a simple composite image as a fallback.
        
        Args:
            person_image_path (str): Path to the person image
            clothing_image_path (str): Path to the clothing image
            fit_analysis (dict): Analysis of the fit
            output_path (str, optional): Path to save the output image
            
        Returns:
            str: Path to the generated image
        """
        try:
            # Load images
            person_img = Image.open(person_image_path).convert("RGBA")
            clothing_img = Image.open(clothing_image_path).convert("RGBA")
            
            # Resize clothing to match person dimensions
            clothing_img = clothing_img.resize(person_img.size)
            
            # Apply transformations based on fit analysis
            overall_fit = fit_analysis.get("overall_fit", "good")
            
            if "loose" in overall_fit:
                # For loose fit, make the clothing slightly larger
                w, h = clothing_img.size
                scale_factor = 1.1
                new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                
                # Resize and center
                resized_clothing = clothing_img.resize((new_w, new_h))
                offset_x, offset_y = (new_w - w) // 2, (new_h - h) // 2
                
                # Create a new transparent image
                temp_img = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
                temp_img.paste(person_img, (offset_x, offset_y), person_img)
                
                # Composite
                result_img = Image.alpha_composite(temp_img, resized_clothing)
                result_img = result_img.crop((offset_x, offset_y, offset_x + w, offset_y + h))
                
            elif "tight" in overall_fit:
                # For tight fit, make the clothing slightly smaller
                w, h = clothing_img.size
                scale_factor = 0.95
                new_w, new_h = int(w * scale_factor), h
                
                # Resize and center
                resized_clothing = clothing_img.resize((new_w, new_h))
                offset_x = (w - new_w) // 2
                
                # Create a new transparent image
                temp_img = Image.new("RGBA", person_img.size, (0, 0, 0, 0))
                temp_img.paste(resized_clothing, (offset_x, 0), resized_clothing)
                
                # Composite
                result_img = Image.alpha_composite(person_img, temp_img)
                
            else:
                # For good fit, simple alpha composite
                result_img = Image.alpha_composite(person_img, clothing_img)
            
            # Convert back to RGB
            result_img = result_img.convert("RGB")
            
            # Add fit annotation
            result_img = self._add_fit_annotation(result_img, fit_analysis)
            
            # Save the result
            if output_path is None:
                output_path = os.path.join(tempfile.gettempdir(), "tryon_result.png")
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result_img.save(output_path)
            
            return output_path
            
        except Exception as e:
            print(f"Error generating fallback try-on image: {str(e)}")
            return None
    
    def _add_fit_annotation(self, image, fit_analysis):
        """
        Add fit annotation to the image.
        
        Args:
            image (PIL.Image): Image to annotate
            fit_analysis (dict): Analysis of the fit
            
        Returns:
            PIL.Image: Annotated image
        """
        # Create a copy of the image
        annotated_img = image.copy()
        
        # Create a drawing context
        draw = ImageDraw.Draw(annotated_img)
        
        # Set font (use default if no font is available)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        
        # Get overall fit
        overall_fit = fit_analysis.get("overall_fit", "unknown")
        
        # Add annotation
        text = f"Fit: {overall_fit.capitalize()}"
        text_color = (255, 255, 255)  # White text
        text_bg = (0, 0, 0, 128)      # Semi-transparent black background
        
        # Get text size
        text_width, text_height = draw.textsize(text, font=font) if hasattr(draw, 'textsize') else (150, 30)
        
        # Draw text background
        draw.rectangle(
            [(10, 10), (10 + text_width + 10, 10 + text_height + 10)],
            fill=text_bg
        )
        
        # Draw text
        draw.text((15, 15), text, fill=text_color, font=font)
        
        return annotated_img

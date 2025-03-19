import os
import json
from PIL import Image
import numpy as np

class SizeAnalyzer:
    """
    Tool for analyzing clothing sizes and fit.
    """
    
    def __init__(self):
        """Initialize the size analyzer."""
        # Size reference data
        self.size_reference = {
            "small": {
                "waist": range(28, 32),  # in inches
                "chest": range(34, 38),
                "shoulders": range(15, 17)
            },
            "medium": {
                "waist": range(32, 36),
                "chest": range(38, 42),
                "shoulders": range(17, 19)
            },
            "large": {
                "waist": range(36, 40),
                "chest": range(42, 46),
                "shoulders": range(19, 21)
            },
            "x-large": {
                "waist": range(40, 44),
                "chest": range(46, 50),
                "shoulders": range(21, 23)
            }
        }
    
    def parse_size(self, size_str):
        """
        Parse size string into standardized format.
        
        Args:
            size_str (str): Size string (e.g., "Large", "44cm waist")
            
        Returns:
            dict: Dictionary with standardized size information
        """
        size_str = size_str.lower()
        size_info = {}
        
        # Check for standard sizes
        if "small" in size_str or "s" == size_str:
            size_info["standard_size"] = "small"
        elif "medium" in size_str or "m" == size_str:
            size_info["standard_size"] = "medium"
        elif "large" in size_str or "l" == size_str:
            size_info["standard_size"] = "large"
        elif "x-large" in size_str or "xl" == size_str:
            size_info["standard_size"] = "x-large"
        
        # Check for measurements
        # Extract numeric measurements with units
        import re
        measurements = re.findall(r'(\d+)(?:\s*)(cm|in|inch|inches)?\s*(waist|chest|shoulders|sleeve|length)?', size_str)
        
        for value, unit, measurement_type in measurements:
            # Convert to inches if necessary
            converted_value = float(value)
            if unit == "cm":
                converted_value = converted_value / 2.54  # Convert cm to inches
            
            if measurement_type:
                size_info[measurement_type] = converted_value
            
        return size_info
    
    def estimate_body_measurements(self, person_image_path):
        """
        Estimate body measurements from an image.
        
        Args:
            person_image_path (str): Path to the person image
            
        Returns:
            dict: Estimated body measurements
        """
        # This would be replaced with actual body measurement estimation
        # For demonstration, we're using placeholder values
        return {
            "waist": 32,  # in inches
            "chest": 38,
            "shoulders": 17
        }
    
    def analyze_fit(self, person_image_path, clothing_size, measurements=None):
        """
        Analyze how a clothing item of a specific size would fit on a person.
        
        Args:
            person_image_path (str): Path to the person image
            clothing_size (str): Clothing size (e.g., "Small", "Medium", "Large")
            measurements (str, optional): Additional measurements (e.g., "44cm waist")
            
        Returns:
            dict: Analysis of the fit
        """
        try:
            # Parse clothing size
            size_info = self.parse_size(clothing_size)
            if measurements:
                additional_size_info = self.parse_size(measurements)
                size_info.update(additional_size_info)
            
            # Estimate body measurements from image
            body_measurements = self.estimate_body_measurements(person_image_path)
            
            # Analyze fit
            fit_analysis = {
                "overall_fit": None,
                "details": {}
            }
            
            # Get standard size reference
            standard_size = size_info.get("standard_size")
            if standard_size and standard_size in self.size_reference:
                reference = self.size_reference[standard_size]
                
                # Check waist fit
                if "waist" in body_measurements:
                    body_waist = body_measurements["waist"]
                    if body_waist < reference["waist"].start:
                        fit_analysis["details"]["waist"] = "loose"
                    elif body_waist > reference["waist"].stop:
                        fit_analysis["details"]["waist"] = "tight"
                    else:
                        fit_analysis["details"]["waist"] = "good"
                
                # Check chest fit
                if "chest" in body_measurements:
                    body_chest = body_measurements["chest"]
                    if body_chest < reference["chest"].start:
                        fit_analysis["details"]["chest"] = "loose"
                    elif body_chest > reference["chest"].stop:
                        fit_analysis["details"]["chest"] = "tight"
                    else:
                        fit_analysis["details"]["chest"] = "good"
                
                # Check shoulder fit
                if "shoulders" in body_measurements:
                    body_shoulders = body_measurements["shoulders"]
                    if body_shoulders < reference["shoulders"].start:
                        fit_analysis["details"]["shoulders"] = "loose"
                    elif body_shoulders > reference["shoulders"].stop:
                        fit_analysis["details"]["shoulders"] = "tight"
                    else:
                        fit_analysis["details"]["shoulders"] = "good"
            
            # Determine overall fit
            if fit_analysis["details"]:
                fit_values = list(fit_analysis["details"].values())
                if all(fit == "good" for fit in fit_values):
                    fit_analysis["overall_fit"] = "good"
                elif all(fit == "loose" for fit in fit_values):
                    fit_analysis["overall_fit"] = "loose"
                elif all(fit == "tight" for fit in fit_values):
                    fit_analysis["overall_fit"] = "tight"
                else:
                    # Mixed fit
                    if fit_values.count("loose") > fit_values.count("tight"):
                        fit_analysis["overall_fit"] = "generally loose"
                    elif fit_values.count("tight") > fit_values.count("loose"):
                        fit_analysis["overall_fit"] = "generally tight"
                    else:
                        fit_analysis["overall_fit"] = "mixed fit"
            
            return fit_analysis
            
        except Exception as e:
            print(f"Error analyzing fit: {str(e)}")
            return {
                "error": str(e)
            }

import os
from crewai import Agent
from langchain.tools import Tool
from tools.image_processor import ImageProcessor
from tools.size_analyzer import SizeAnalyzer
from models.virtual_tryon import VirtualTryOn

def create_fitting_room_agent(image_processor: ImageProcessor, size_analyzer: SizeAnalyzer):
    """
    Creates the Virtual Fitting Room agent that handles clothing visualization.
    
    Args:
        image_processor: Tool for image processing
        size_analyzer: Tool for analyzing clothing sizes
        
    Returns:
        Agent: The configured fitting room agent
    """
    # Initialize the virtual try-on model
    virtual_tryon = VirtualTryOn()
    
    # Create tools for the agent
    process_image_tool = Tool(
        name="Process Image",
        func=image_processor.process_images,
        description="Process person and clothing images for virtual try-on."
    )
    
    analyze_size_tool = Tool(
        name="Analyze Size",
        func=size_analyzer.analyze_fit,
        description="Analyze how a clothing item of a specific size would fit on a person."
    )
    
    virtual_tryon_tool = Tool(
        name="Virtual Try-On",
        func=virtual_tryon.generate_tryon_image,
        description="Generate a virtual try-on image of a person wearing a clothing item."
    )
    
    # Create the fitting room agent
    agent = Agent(
        role="Virtual Fitting Room Specialist",
        goal="Create realistic visualizations of people wearing specific clothing items with accurate size representation",
        backstory="""
        You are an expert in virtual fashion and computer vision. You've worked with top fashion 
        brands to create virtual fitting solutions. Your specialty is in creating realistic 
        visualizations of clothing on people while accurately representing how different 
        sizes would fit on different body types.
        """,
        verbose=True,
        allow_delegation=False,
        tools=[process_image_tool, analyze_size_tool, virtual_tryon_tool]
    )
    
    return agent

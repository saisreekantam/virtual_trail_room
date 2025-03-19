import os
from dotenv import load_dotenv
from crewai import Crew, Agent, Task
from agents.fitting_room_agent import create_fitting_room_agent
from agents.style_consultant_agent import create_style_consultant_agent
from tools.image_processor import ImageProcessor
from tools.size_analyzer import SizeAnalyzer
import argparse

# Load environment variables
load_dotenv()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Virtual Fitting Room with Style Consultant')
    parser.add_argument('--person_image', type=str, required=True, help='Path to person image')
    parser.add_argument('--clothing_image', type=str, required=True, help='Path to clothing image')
    parser.add_argument('--size', type=str, required=True, help='Clothing size (e.g., Small, Medium, Large)')
    parser.add_argument('--measurements', type=str, help='Additional measurements (e.g., "44cm waist")')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save output images')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tools
    image_processor = ImageProcessor()
    size_analyzer = SizeAnalyzer()
    
    # Create agents
    fitting_room_agent = create_fitting_room_agent(image_processor, size_analyzer)
    style_consultant_agent = create_style_consultant_agent()
    
    # Define tasks
    fitting_task = Task(
        description=f"""
        Analyze the person image at {args.person_image} and the clothing image at {args.clothing_image}.
        The clothing size is {args.size} with measurements {args.measurements if args.measurements else 'not specified'}.
        Embed the clothing onto the person and create a realistic visualization.
        Save the result to {os.path.join(args.output_dir, 'fitted_image.png')}.
        Analyze how the clothing fits based on the person's body type and the clothing size.
        """,
        agent=fitting_room_agent,
        expected_output="A description of the fitting process and the path to the generated image."
    )
    
    consultation_task = Task(
        description=f"""
        Analyze the fitted image at {os.path.join(args.output_dir, 'fitted_image.png')}.
        Evaluate how the clothing looks on the person.
        Comment on the fit (small, tight, good, loose, etc.).
        Provide fashion advice and styling recommendations.
        """,
        agent=style_consultant_agent,
        expected_output="A detailed style consultation report."
    )
    
    # Create the crew
    crew = Crew(
        agents=[fitting_room_agent, style_consultant_agent],
        tasks=[fitting_task, consultation_task],
        verbose=True
    )
    
    # Start the crew
    result = crew.kickoff()
    
    print("\n======= Virtual Fitting Room Results =======")
    print(result)
    print("===========================================")

if __name__ == "__main__":
    main()

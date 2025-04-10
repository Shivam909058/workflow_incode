from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
import sys
import io

# Simple context manager to store and retrieve content without requiring user input
class WorkflowContext:
    def __init__(self):
        self.data = {}
    
    def store(self, key, value):
        self.data[key] = value
    
    def get(self, key, default=None):
        return self.data.get(key, default)

# Capture stdout function to avoid requiring user to copy-paste
def capture_output(func, *args, **kwargs):
    """Capture output from a function call"""
    old_stdout = sys.stdout
    capturer = io.StringIO()
    sys.stdout = capturer
    
    try:
        func(*args, **kwargs)
    finally:
        sys.stdout = old_stdout
    
    output = capturer.getvalue()
    capturer.close()
    return output

# Define our content generation agent with web search capabilities
content_generator = Agent(
    name="Content Generator",
    role="Generate LinkedIn content based on topic and audience",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    description="You are a professional content creator for LinkedIn who researches topics and creates engaging posts.",
    instructions=[
        "Research current trends and information on the web to make the content relevant and data-driven.",
        "Format your response with an attention-grabbing headline, well-structured main content (max 3000 characters), and 3 relevant hashtags.",
        "Ensure the content is professional, engaging, and appropriate for LinkedIn.",
        "Include statistics or examples that support your points.",
        "Use formatting (bold, bullets, etc.) to improve readability.",
        "The first two lines should be compelling as they appear in the LinkedIn feed preview."
    ],
    show_tool_calls=True,
    markdown=True,
)

# Define our content refinement agent
content_refiner = Agent(
    name="Content Refiner",
    role="Refine and improve LinkedIn content",
    model=OpenAIChat(id="gpt-4o"),
    description="You are a professional content editor for LinkedIn who improves and polishes content.",
    instructions=[
        "Refine the given content to make it more engaging, professional, and impactful.",
        "Ensure the content is concise, well-structured, and appropriate for LinkedIn's format.",
        "Maintain the core message while improving clarity, tone, and impact.",
        "Format the response with clear sections: headline, content body, and hashtags.",
        "Apply LinkedIn best practices: use emojis sparingly, create white space, use bullets or numbers.",
        "Ensure the first two lines are compelling as they appear in the LinkedIn feed preview."
    ],
    show_tool_calls=False,
    markdown=True,
)

# Create our team for content operations
linkedin_team = Team(
    name="LinkedIn Content Team",
    mode="coordinate",  # Coordinate mode for team collaboration
    members=[content_generator, content_refiner],
    model=OpenAIChat(id="gpt-4o"),
    description="A team that creates and refines professional LinkedIn content.",
    instructions=[
        "Follow LinkedIn best practices", 
        "Ensure content is engaging and professional",
        "Format content appropriately for LinkedIn's platform",
        "Maintain and update the context with relevant information",
        "Remember user feedback and incorporate it into the workflow",
        "The Content Generator should handle initial content creation and research",
        "The Content Refiner should focus on optimizing and polishing the content"
    ],
    show_tool_calls=True,
    markdown=True,
    show_members_responses=True,
    enable_agentic_context=True,
    share_member_interactions=True,
    enable_team_history=True,
    num_of_interactions_from_history=5,
)

def main():
    """
    Main function to run the LinkedIn content posting workflow.
    """
    # Initialize workflow context
    context = WorkflowContext()
    
    print("\n=== LinkedIn Content Posting Workflow ===\n")
    
    # Step 1: Content Idea Input
    print("== CONTENT IDEA INPUT ==")
    topic = input("What topic would you like to post about? ")
    audience = input("Who is your target audience? ")
    
    context.store("topic", topic)
    context.store("audience", audience)
    
    # Step 2: Content Generation using the team
    print("\n== CONTENT GENERATION ==")
    print("Generating content based on your topic and audience...")
    
    generation_prompt = f"""
    Task: Create a professional LinkedIn post about {topic} for {audience}.
    
    This task should be handled by the Content Generator, who should:
    1. Research current trends and data about {topic}
    2. Create an attention-grabbing headline
    3. Write an engaging introduction that hooks the reader
    4. Develop 3-5 key points with supporting evidence
    5. Include a clear call-to-action
    6. Add 3 relevant hashtags
    
    The post should be informative, professional, and optimized for LinkedIn's algorithm.
    """
    
    linkedin_team.print_response(generation_prompt, stream=True)
    
    # Content Review Loop - keep refining until user approves
    content_approved = False
    while not content_approved:
        print("\n== CONTENT REVIEW ==")
        feedback = input("Is this content good? (yes/no): ")
        
        if feedback.lower() in ["yes", "y"]:
            content_approved = True
        else:
            user_feedback = input("What specific feedback do you have for improving the content? ")
            context.store("user_feedback", user_feedback)
            
            print("\n== REGENERATING CONTENT ==")
            print("Regenerating content based on your feedback...")
            
            regeneration_prompt = f"""
            Task: Improve the LinkedIn post about {topic} for {audience} based on user feedback.
            
            User Feedback: {user_feedback}
            
            The Content Generator should address all the feedback points while maintaining a professional and engaging LinkedIn post format.
            """
            
            linkedin_team.print_response(regeneration_prompt, stream=True)
    
    # Step 4: Content Refinement
    print("\n== CONTENT REFINEMENT ==")
    print("Refining and enhancing your content...")
    
    refinement_prompt = f"""
    Task: Refine and enhance the LinkedIn post about {topic} for {audience}.
    
    This task should be handled by the Content Refiner, who should:
    1. Improve the headline to make it more attention-grabbing
    2. Enhance the first two sentences (they appear in LinkedIn feed preview)
    3. Optimize structure with appropriate spacing and formatting
    4. Add strategic emojis if appropriate
    5. Strengthen the call-to-action
    6. Review and improve the hashtags for better reach
    
    The refined post should be more engaging, professional, and impactful.
    """
    
    linkedin_team.print_response(refinement_prompt, stream=True)
    
    # Refinement Approval Loop - keep refining until user approves
    refinement_approved = False
    while not refinement_approved:
        print("\n== REFINEMENT REVIEW ==")
        approval = input("Do you approve this refined content? (yes/no): ")
        
        if approval.lower() in ["yes", "y"]:
            refinement_approved = True
        else:
            final_feedback = input("What specific feedback do you have for the refinement? ")
            context.store("final_feedback", final_feedback)
            
            print("\n== REFINING CONTENT AGAIN ==")
            print("Refining content based on your feedback...")
            
            additional_refinement_prompt = f"""
            Task: Make additional refinements to the LinkedIn post based on user feedback.
            
            User Feedback: {final_feedback}
            
            The Content Refiner should address all the feedback points while maintaining the professional and engaging nature of the post.
            """
            
            linkedin_team.print_response(additional_refinement_prompt, stream=True)
    
    # Final Approval Loop - ensure user is completely satisfied
    final_approved = False
    while not final_approved:
        print("\n== FINAL APPROVAL ==")
        final_approval = input("Is this content perfectly okay to post on LinkedIn? (yes/no): ")
        
        if final_approval.lower() in ["yes", "y"]:
            final_approved = True
        else:
            last_feedback = input("What final changes are needed before posting? ")
            
            print("\n== MAKING FINAL CHANGES ==")
            print("Implementing your final changes...")
            
            final_changes_prompt = f"""
            Task: Make these final changes to the LinkedIn post before publication:
            
            User Feedback: {last_feedback}
            
            Create the absolutely final version that is ready for LinkedIn publishing.
            """
            
            linkedin_team.print_response(final_changes_prompt, stream=True)
    
    # Step 6: Post to LinkedIn
    print("\n== POST TO LINKEDIN ==")
    print("Preparing your post for LinkedIn...")
    
    final_format_prompt = """
    Task: Provide the complete final version of the LinkedIn post in a clean, ready-to-publish format.
    
    Format the post exactly as it should appear when published on LinkedIn, with proper spacing, formatting, and hashtags.
    """
    
    print("\nYour post is ready to be published to LinkedIn:")
    print("=" * 50)
    linkedin_team.print_response(final_format_prompt, stream=True)
    print("=" * 50)
    
    print("\nIn a real implementation, this would be posted to LinkedIn using their API.")
    print("\nWorkflow completed successfully!")

if __name__ == "__main__":
    main()


"""
This code creates a LinkedIn content posting workflow with the following steps:
-> Content Idea Input: Takes user input for content topic and target audience
-> Content Generation: Uses an Agno agent with web search capabilities to generate content
-> Content Review: Gets user feedback and either proceeds or regenerates content
-> Content Refinement: Uses another agent to refine and enhance the content
-> Final Approval: Gets final user approval before posting
-> Post to LinkedIn: Prepares the post for LinkedIn (with placeholders for actual API integration)

"""     
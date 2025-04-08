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
        "Use formatting (bold, bullets, etc.) to improve readability."
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
        "Remember user feedback and incorporate it into the workflow"
    ],
    show_tool_calls=True,
    markdown=True,
    show_members_responses=True,
    enable_agentic_context=True,  # Enable agentic context for better memory
    share_member_interactions=True,  # Enable member interaction sharing
    enable_team_history=True,     # Enable team history for conversation context
    num_of_interactions_from_history=5,  # Remember the last 5 interactions
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
    
    # Step 2: Content Generation - use the team directly now
    print("\n== CONTENT GENERATION ==")
    print("Generating content based on your topic and audience...")
    
    # Construct generation prompt
    generation_prompt = f"""
    Create a professional LinkedIn post about {topic} for {audience}.
    
    Requirements:
    1. Research current trends and data about {topic}
    2. Create an attention-grabbing headline
    3. Write an engaging introduction that hooks the reader
    4. Develop 3-5 key points with supporting evidence
    5. Include a clear call-to-action
    6. Add 3 relevant hashtags
    
    The post should be informative, professional, and optimized for LinkedIn's algorithm.
    """
    
    # With OpenAI, we can now use the team's coordinate mode directly
    print("\nTeam Response:")
    linkedin_team.print_response(generation_prompt, stream=True)
    
    # Capture content for internal use
    generated_content = capture_output(content_generator.print_response, generation_prompt, stream=False)
    context.store("generated_content", generated_content)
    
    # Step 3: Content Review
    print("\n== CONTENT REVIEW ==")
    feedback = input("Is this content good? (yes/no): ")
    
    if feedback.lower() not in ["yes", "y"]:
        user_feedback = input("What specific feedback do you have for improving the content? ")
        context.store("user_feedback", user_feedback)
        
        print("\n== REGENERATING CONTENT ==")
        print("Regenerating content based on your feedback...")
        
        # Construct regeneration prompt with feedback
        regeneration_prompt = f"""
        Please improve the LinkedIn post based on this feedback: {user_feedback}
        
        The post should be about {topic} for {audience}.
        """
        
        # Since we're using OpenAI, we can use the team with full context
        print("\nRegenerated Content:")
        linkedin_team.print_response(regeneration_prompt, stream=True)
    
    # Step 4: Content Refinement
    print("\n== CONTENT REFINEMENT ==")
    print("Refining and enhancing your content...")
    
    # Construct refinement prompt - with team history, it should have access to previous content
    refinement_prompt = f"""
    Take the LinkedIn post about {topic} for {audience} and refine it further.
    
    Refinement Requirements:
    1. Improve the headline to make it more attention-grabbing
    2. Enhance the first two sentences (they appear in LinkedIn feed preview)
    3. Optimize structure with appropriate spacing and formatting
    4. Add strategic emojis if appropriate
    5. Strengthen the call-to-action
    6. Review and improve the hashtags for better reach
    """
    
    # Let the team handle the refinement 
    print("\nRefined Content:")
    linkedin_team.print_response(refinement_prompt, stream=True)
    
    # Step 5: Final Approval
    print("\n== FINAL APPROVAL ==")
    approval = input("Do you approve this content for posting? (yes/no): ")
    
    if approval.lower() not in ["yes", "y"]:
        final_feedback = input("What specific feedback do you have for the refinement? ")
        context.store("final_feedback", final_feedback)
        
        print("\n== REFINING CONTENT AGAIN ==")
        print("Refining content based on your feedback...")
        
        # With OpenAI, we can use simpler prompts as the context is maintained
        final_refinement_prompt = f"""
        Please make these final improvements to the LinkedIn post: {final_feedback}
        
        This is the final version that will be published.
        """
        
        # Use the team with its context
        print("\nFinal Refined Content:")
        linkedin_team.print_response(final_refinement_prompt, stream=True)
    
    # Step 6: Post to LinkedIn
    print("\n== POST TO LINKEDIN ==")
    print("Preparing your post for LinkedIn...")
    
    # Get the final formatted version
    final_format_prompt = "Please provide the complete final version of the LinkedIn post in a clean, ready-to-publish format."
    
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
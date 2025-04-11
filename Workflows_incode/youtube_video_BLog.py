# youtube_video_blog.py - YouTube to  Blog Post Workflow

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude
from agno.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.youtube import YouTubeTools
import sys
import io
import os

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

# YouTube Content Extraction Agent
youtube_extractor = Agent(
    name="YouTube Extractor",
    role="Extract comprehensive content from YouTube videos",
    model=OpenAIChat(id="gpt-4o"),  # Changed from Claude to GPT-4o
    tools=[YouTubeTools()],
    description="You are a professional content extractor who pulls key information from YouTube videos.",
    instructions=[
        "Extract the main themes, key points, and valuable insights from YouTube videos",
        "Capture direct quotes that are impactful or noteworthy",
        "Identify the structure and flow of the video content",
        "Note any statistics, examples, or case studies mentioned",
        "Summarize the content in a comprehensive way that retains all important information",
        "Pay special attention to the expertise and authority signals in the content"
    ],
    show_tool_calls=True,
    markdown=True,
)

# Keyword Research Agent
keyword_researcher = Agent(
    name="Keyword Researcher",
    role="Research optimal keywords for content visibility",
    model=OpenAIChat(id="gpt-4o"),  # Changed from Claude to GPT-4o
    tools=[DuckDuckGoTools()],
    description="You are an SEO expert who identifies valuable keywords and content opportunities.",
    instructions=[
        "Research current trending keywords related to the video topic",
        "Identify high-volume, low-competition keywords",
        "Analyze competitor content to find content gaps",
        "Suggest primary and secondary keywords for content optimization",
        "Recommend semantic keywords and related terms to enhance SEO",
        "Provide insights on search intent for the identified keywords",
        "Consider keywords that work well across multiple content platforms"
    ],
    show_tool_calls=True,
    markdown=True,
)

# Content Writer Agent
content_writer = Agent(
    name="Content Writer",
    role="Create engaging, optimized blog content",
    model=OpenAIChat(id="gpt-4o"),  # Changed from Claude to GPT-4o
    tools=[DuckDuckGoTools()],
    description="You are a professional content writer who transforms video content into engaging multi-platform blog posts.",
    instructions=[
        "Create compelling headlines and introductions that capture attention",
        "Transform video content into well-structured, readable blog format",
        "Maintain the original message while enhancing clarity and flow",
        "Add relevant context and background information when needed",
        "Develop a consistent voice and tone appropriate for blogs and article platforms",
        "Ensure content is comprehensive, informative, and valuable to readers",
        "Naturally incorporate primary and secondary keywords",
        "Create content that can work across multiple publishing platforms"
    ],
    show_tool_calls=True,
    markdown=True,
)

# Content Refiner Agent
content_refiner = Agent(
    name="Content Refiner",
    role="Refine and enhance raw blog content",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    description="You are a professional content refiner who enhances raw content into polished, engaging material for multiple platforms.",
    instructions=[
        "Take raw blog content and enhance its structure, flow, and readability",
        "Improve transitions between sections for seamless reading experience",
        "Generate comprehensive, in-depth content (minimum 2500-3000 words)",
        "Add depth and nuance to explanations with detailed research",
        "Include extensive examples, case studies, and supporting evidence",
        "Ensure content speaks directly to the target audience's needs and interests",
        "Strengthen arguments with additional context and supporting points",
        "Balance informational content with engaging narrative elements",
        "Incorporate detailed statistics and data to support claims",
        "Maintain SEO optimization while improving overall quality",
        "Ensure content works well across different publishing platforms (not just LinkedIn)",
        "When feedback is received, make substantial and meaningful changes",
        "Provide detailed actionable advice, not just general information"
    ],
    show_tool_calls=True,
    markdown=True,
)

# SEO Optimizer Agent
seo_optimizer = Agent(
    name="SEO Optimizer",
    role="Optimize content for search engines",
    model=OpenAIChat(id="gpt-4o"),  # Changed from Claude to GPT-4o
    description="You are an SEO expert who optimizes content for maximum visibility and engagement across multiple platforms.",
    instructions=[
        "Optimize headline and subheadings with relevant keywords",
        "Ensure proper keyword density and placement throughout the content",
        "Add semantic markup recommendations (header tags, meta descriptions)",
        "Improve readability with appropriate paragraphing and sentence structure",
        "Suggest internal and external linking opportunities",
        "Optimize for featured snippets and rich results",
        "Ensure content meets E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness) criteria",
        "Provide platform-specific SEO recommendations for WordPress, Medium, community forums, and other platforms"
    ],
    show_tool_calls=False,
    markdown=True,
)

# Final Editor Agent
final_editor = Agent(
    name="Final Editor",
    role="Polish and finalize content for publication",
    model=OpenAIChat(id="gpt-4o"),  # Changed from Claude to GPT-4o
    description="You are a professional editor who polishes content to perfection for publishing across multiple platforms.",
    instructions=[
        "Check for grammatical errors and improve sentence structure",
        "Enhance flow and coherence between paragraphs and sections",
        "Ensure proper formatting for multiple publishing platforms",
        "Add compelling call-to-action elements",
        "Create an eye-catching title and meta description",
        "Ensure consistent tone and voice throughout",
        "Format content with appropriate spacing, bullets, and emphasis",
        "Provide platform-specific formatting recommendations for WordPress, Medium, community forums, and social media"
    ],
    show_tool_calls=False,
    markdown=True,
)

blog_team = Team(
    name="YouTube to Multi-Platform Blog Team",
    mode="coordinate",
    members=[youtube_extractor, keyword_researcher, content_writer, content_refiner, seo_optimizer, final_editor],
    model=OpenAIChat(id="gpt-4o"),  # Team coordinator using GPT-4o
    description="A team that converts YouTube videos into high-quality blog posts for multiple publishing platforms.",
    instructions=[
        "Extract valuable content from YouTube videos",
        "Research and incorporate relevant keywords",
        "Create engaging, well-structured blog content",
        "Refine raw content into polished material",
        "Optimize content for both SEO and readability",
        "Ensure the final product is polished and ready for multiple publishing platforms",
        "Maintain the original value and message of the video",
        "Coordinate smoothly between extraction, writing, refinement, and optimization phases",
        "Consider the specific requirements of different publishing platforms"
    ],
    show_tool_calls=True,
    markdown=True,
    show_members_responses=True,
    enable_agentic_context=True,
    share_member_interactions=True,
    enable_team_history=True,
    num_of_interactions_from_history=15,  # Increased history for better context
)

# Modified SEO optimization prompt (to be used after content refinement)
seo_optimization_prompt = f"""
Task: Optimize the refined blog post for maximum search visibility, readability, and E-E-A-T signals across multiple publishing platforms.

This task should be handled by the SEO Optimizer, who should provide the following structured output:

1. TITLE OPTIMIZATION:
   * Analyze the current title(s) and provide an improved SEO title
   * Ensure primary keyword appears near the beginning
   * Keep title between 50-60 characters
   * Balance keyword optimization with click-worthiness
   * Explain the specific improvements made

2. META DESCRIPTION:
   * Create a compelling meta description under 155 characters
   * Include primary keyword and a clear value proposition
   * Add a subtle call-to-action element
   * Ensure it accurately represents the content

3. HEADING STRUCTURE OPTIMIZATION:
   * Analyze and optimize the H1, H2, and H3 structure
   * Ensure proper keyword integration in headings
   * Verify logical hierarchy and content flow
   * Provide specific rewrites for any suboptimal headings

4. CONTENT BODY OPTIMIZATION:
   * Identify opportunities to improve keyword placement
   * Highlight paragraphs needing better keyword integration
   * Suggest LSI keyword additions for specific sections
   * Recommend sentence structure improvements for readability
   * Ensure proper keyword density (without overstuffing)

5. E-E-A-T ENHANCEMENT:
   * Identify opportunities to strengthen Experience signals
   * Suggest additions to reinforce Expertise indicators
   * Recommend Authoritativeness improvements
   * Enhance Trustworthiness through specific content additions
   * Provide exact wording for these enhancements

6. FEATURED SNIPPET OPTIMIZATION:
   * Identify 1-3 opportunities for featured snippet targeting
   * Provide structured content formats (lists, tables, definitions)
   * Rewrite specific sections to maximize snippet potential
   * Explain the reasoning behind each optimization

7. INTERNAL/EXTERNAL LINKING STRATEGY:
   * Recommend 3-5 specific places for external authority links
   * Suggest anchor text optimization for these links
   * Provide rationale for each linking opportunity

8. TECHNICAL SEO RECOMMENDATIONS:
   * Suggest schema markup opportunities (if applicable)
   * Recommend image alt text for any visual elements
   * Address any content accessibility improvements

9. PLATFORM-SPECIFIC OPTIMIZATIONS:
   * WordPress-specific SEO recommendations (Yoast or similar plugins)
   * Medium-specific optimization strategies
   * Social media sharing optimization
   * Community forum engagement optimization
   * Recommendations for adapting SEO across different platforms

Provide specific, implementable recommendations with exact wording where appropriate. Maintain the voice and expertise level of the refined content while enhancing its search visibility and user experience across multiple publishing platforms.
"""

def main():
    """
    Main function to run the YouTube to Multi-Platform Blog workflow.
    """
    # Initialize workflow context
    context = WorkflowContext()
    
    print("\n=== YouTube to Multi-Platform Blog Generator ===\n")
    
    # Step 1: User Input
    print("== VIDEO INPUT ==")
    video_url = input("Please enter the YouTube video URL: ")
    target_audience = input("Who is the target audience for this blog post? ")
    platform_preference = input("What publishing platforms are you targeting? (e.g., WordPress, Medium, community forums): ")
    
    context.store("video_url", video_url)
    context.store("target_audience", target_audience)
    context.store("platform_preference", platform_preference)
    
    # Step 2: Video Content Extraction
    print("\n== EXTRACTING VIDEO CONTENT ==")
    print("Analyzing and extracting content from the YouTube video...")
    
    extraction_prompt = f"""
    Task: Extract comprehensive content from the YouTube video at {video_url}.
    
    This task should be handled by the YouTube Extractor, who should provide the following structured output:
    
    1. VIDEO OVERVIEW:
       * Title and creator of the video
       * Publication date and duration
       * Primary topic and purpose of the content
       * Target audience of the original video
    
    2. CONTENT STRUCTURE:
       * Provide a detailed breakdown of the video's structure with timestamps
       * Identify the main sections/chapters of the video
       * Note how ideas progress and connect throughout the video
    
    3. KEY POINTS & INSIGHTS:
       * Extract ALL major points, arguments, and insights presented
       * List each point as a separate bullet with supporting details
       * Include any data, statistics, or research mentioned (with specific numbers)
    
    4. DIRECT QUOTES:
       * Transcribe 5-10 impactful direct quotes from the speaker(s)
       * Include context for each quote and timestamp if possible
       * Note any particularly powerful or unique phrasing
    
    5. EXAMPLES & CASE STUDIES:
       * Detail any examples, stories, or case studies shared
       * Explain how these examples support the main arguments
       * Note any real-world applications demonstrated
    
    6. VISUAL ELEMENTS:
       * Describe any important visual aids, demonstrations, or graphics shown
       * Explain what these visuals illustrate and their significance
       * Note any text on slides or important visual information
    
    7. EXPERT CREDIBILITY:
       * Identify credentials, experience, or authority signals mentioned
       * Note any references to research, books, or external sources
       * Extract any methodology explanations or proof points
    
    8. COMPREHENSIVE SUMMARY:
       * Provide a 300-500 word summary of the entire video content
       * Capture the essence, main message, and unique value of the video
       * Include any conclusion or call-to-action from the video
    
    The extraction must be thorough, accurate, and retain ALL valuable information for blog conversion.
    """
    
    blog_team.print_response(extraction_prompt, stream=True)
    
    # Step 3: Keyword Research
    print("\n== KEYWORD RESEARCH ==")
    print("Researching relevant keywords and SEO opportunities...")
    
    keyword_prompt = f"""
    Task: Conduct comprehensive keyword research for a blog post based on the extracted video content.
    
    This task should be handled by the Keyword Researcher, who should provide the following structured output:
    
    1. PRIMARY KEYWORD ANALYSIS:
       * Identify 3-5 high-value primary keywords related to the video topic
       * For each primary keyword, provide:
         - Monthly search volume (if available)
         - Competition level (high/medium/low)
         - Search intent (informational, commercial, navigational, transactional)
         - Why this keyword matches our content and audience
    
    2. SECONDARY KEYWORDS:
       * List 10-15 secondary keywords and phrases to incorporate
       * Group these by subtopic or content section
       * Note which secondary keywords should appear in headings
    
    3. SEMANTIC/LSI KEYWORDS:
       * Provide 15-20 semantic keywords and related terms
       * Explain how these terms enhance topical authority
       * Suggest natural placement opportunities in the content
    
    4. TRENDING TERMS & HASHTAGS:
       * Identify 5-7 trending terms or hashtags in this topic area
       * Note which LinkedIn and twitter and facebook and instagram hashtags will maximize visibility
       * Suggest industry-specific terminology to demonstrate expertise
    
    5. COMPETITOR KEYWORD ANALYSIS:
       * List 3-5 top-ranking content pieces on similar topics
       * Identify keywords they're targeting that we should include
       * Note any content gaps or opportunities they've missed
    
    6. KEYWORD PLACEMENT STRATEGY:
       * Provide specific recommendations for:
         - Title/headline keyword optimization
         - Subheading keyword placement
         - Introduction and conclusion keywords
         - Natural keyword integration strategies for readability
    
    7. SEARCH INTENT INSIGHTS:
       * Analyze what users are specifically looking for with these keywords
       * Explain how our content should address these specific needs
       * Suggest content structure to match search intent patterns
    
    The target audience is: {target_audience}
    
    Your analysis should prioritize keywords that will resonate with this audience while maximizing visibility and engagement on LinkedIn and other platforms to post this blog like facebook and instagram and twitter medium and wordpress.
    """
    
    blog_team.print_response(keyword_prompt, stream=True)
    
    # Step 4: Initial Content Creation
    print("\n== CREATING INITIAL CONTENT ==")
    print("Generating initial blog post draft...")
    
    content_prompt = f"""
    Task: Create a comprehensive, in-depth blog post draft based on the video content and keyword research.

    This task should be handled by the Content Writer, who should provide the following structured output:

    1. HEADLINE OPTIONS (provide 5-7):
       * Create compelling, keyword-rich headlines under 60 characters
       * Each headline should include primary keywords while grabbing attention
       * Vary approaches (question, how-to, listicle, statement, etc.)
       * Ensure headlines work well across multiple publishing platforms

    2. INTRODUCTION (300-400 words):
       * Hook the reader with a powerful opening statement or question
       * Establish relevance to the reader's challenges or interests
       * Include primary keyword naturally within first 100 words
       * Set expectations for what the post will cover
       * End with a clear transition to the main content

    3. MAIN BODY (structured in clear sections):
       Create 6-8 comprehensive sections with:
       * Strong subheading with secondary keyword integration
       * Opening paragraph that establishes the section's purpose
       * At least 300-400 words per section with detailed information
       * Key points, insights, and supporting evidence from the video
       * Relevant examples, case studies, or applications
       * Multiple data points or statistics for each main claim
       * Expert quotes from the video with proper attribution
       * Visual element recommendations (charts, images, etc.)
       * Transition to the next section

    4. PRACTICAL APPLICATION (400-500 words):
       * Create a dedicated section showing how to apply these insights
       * Include detailed step-by-step guidance or actionable frameworks
       * Add value beyond what was explicitly stated in the video
       * Incorporate relevant industry examples or use cases
       * Create a helpful checklist or process outline for implementation

    5. EXPERT PERSPECTIVE (300-400 words):
       * Highlight the expertise demonstrated in the video
       * Connect insights to broader industry trends or research
       * Establish why these insights matter in today's context
       * Naturally incorporate authority-building semantic keywords
       * Include additional expert opinions from web research

    6. CONCLUSION (200-300 words):
       * Summarize the key takeaways without simply repeating points
       * Reinforce the primary value proposition or main insight
       * Include a forward-looking statement or prediction
       * End with a compelling call-to-action

    7. ENGAGEMENT ELEMENTS:
       * Include 3-5 thoughtful questions to prompt comments
       * Add multiple clear, specific calls-to-action for readers
       * Suggest relevant tags/categories for different platforms

    The content should be informative, engaging, and expertly optimized for the target audience: {target_audience}

    Maintain a professional yet conversational tone, use varied sentence structures, and ensure all content provides unique value while naturally incorporating your keyword strategy. The total content length should be 2500-3000 words minimum. The content should be adaptable for publishing on WordPress, Medium, community forums, and other content platforms without being specific to any single platform.
    """
    
    blog_team.print_response(content_prompt, stream=True)
    
    # Content Review Loop - keep refining until user approves
    content_approved = False
    while not content_approved:
        print("\n== INITIAL CONTENT REVIEW ==")
        feedback = input("Is this initial content draft good? (yes/no): ")
        
        if feedback.lower() in ["yes", "y"]:
            content_approved = True
        else:
            user_feedback = input("What specific feedback do you have for improving the content? ")
            context.store("user_feedback", user_feedback)
            
            print("\n== REGENERATING CONTENT ==")
            print("Regenerating content based on your feedback...")
            
            regeneration_prompt = f"""
            Task: Completely redesign and improve the blog post based on user feedback.
            
            User Feedback: {user_feedback}
            
            The Content Writer should:
            1. Make SUBSTANTIAL changes to address ALL feedback points
            2. Significantly expand content length to 2500-3000 words minimum
            3. Add at least 3-5 new sections with detailed information not included before
            4. Include extensive research, statistics, and examples
            5. Make the content universally applicable to multiple platforms (NOT LinkedIn-specific)
            6. Make sure every section is thoroughly developed with actionable insights
            7. Restructure content completely if necessary to address feedback
            8. Focus on providing specific, practical advice rather than general statements
            
            Provide a completely rewritten draft that is substantially different from the previous version.
            """
            
            blog_team.print_response(regeneration_prompt, stream=True)
    
    # NEW STEP: Content Refinement (after initial content creation is approved)
    print("\n== CONTENT REFINEMENT ==")
    print("Refining and enhancing the initial blog draft...")
    
    content_refinement_prompt = f"""
    Task: Refine and enhance the initial blog content to dramatically improve quality, depth, and engagement.

    This task should be handled by the Content Refiner, who should provide the following structured improvements:

    1. NARRATIVE STRUCTURE ENHANCEMENT:
       * Improve the overall flow and logical progression
       * Strengthen the central narrative thread connecting all sections
       * Add compelling transitions between paragraphs and sections
       * Create a more cohesive introduction-to-conclusion journey
       * Ensure each section builds upon previous insights

    2. CONTENT DEPTH ENRICHMENT:
       * Expand on complex or important concepts with additional context
       * Add nuanced explanations for industry-specific terminology
       * Provide more detailed examples that illustrate key points
       * Strengthen arguments with additional supporting evidence
       * Fill in any logical gaps in the presented information

    3. ENGAGEMENT AMPLIFICATION:
       * Transform passive language into active, engaging phrasing
       * Add compelling questions or thought-provoking statements
       * Incorporate narrative techniques to maintain reader interest
       * Create psychological hooks throughout the content
       * Strengthen emotional resonance while maintaining professionalism

    4. AUDIENCE ALIGNMENT:
       * Refine language and examples to better connect with {target_audience}
       * Address specific pain points or challenges faced by this audience
       * Incorporate terminology and references familiar to this audience
       * Enhance the practical value proposition for these specific readers
       * Anticipate and address potential questions or objections

    5. CLARITY & READABILITY OPTIMIZATION:
       * Simplify overly complex sentences without losing sophistication
       * Break down complicated concepts into digestible components
       * Improve paragraph structure for optimal comprehension
       * Enhance logical flow between ideas within paragraphs
       * Ensure consistent terminology and clear referencing

    6. VALUE PROPOSITION STRENGTHENING:
       * Clarify and amplify the unique insights provided
       * Strengthen the practical applications of the presented information
       * Enhance the actionable takeaways and their implementation guidance
       * Connect content value more explicitly to reader benefits
       * Elevate the perceived expertise level throughout

    7. TONE & VOICE REFINEMENT:
       * Ensure consistent professional yet conversational tone
       * Balance authoritative expertise with approachable language
       * Fine-tune the personality and voice to match blogging best practices
       * Adjust formality level appropriately for the target audience
       * Maintain appropriate confidence level in assertions and advice

    8. CROSS-PLATFORM ADAPTABILITY:
       * Ensure the structure works well across multiple publishing platforms
       * Consider how the content will appear on WordPress, Medium, and other sites
       * Add elements that enhance readability on mobile devices
       * Ensure formatting is adaptable for different platform requirements
       * Consider platform-specific engagement patterns

    The refined content should represent a significant quality improvement over the initial draft while preserving all valuable insights and SEO elements. Focus on making the content more compelling, valuable, and professionally polished for {target_audience} across multiple publishing platforms.
    """
    
    blog_team.print_response(content_refinement_prompt, stream=True)
    
    # Content Refinement Review Loop
    refinement_approved = False
    while not refinement_approved:
        print("\n== REFINEMENT REVIEW ==")
        refinement_feedback = input("Is this refined content good? (yes/no): ")
        
        if refinement_feedback.lower() in ["yes", "y"]:
            refinement_approved = True
        else:
            refinement_user_feedback = input("What specific feedback do you have for improving the refinement? ")
            context.store("refinement_user_feedback", refinement_user_feedback)
            
            print("\n== REFINING CONTENT AGAIN ==")
            print("Re-refining content based on your feedback...")
            
            re_refinement_prompt = f"""
            Task: Fundamentally transform the blog post based on new user feedback.
            
            User Feedback on Refinement: {refinement_user_feedback}
            
            The Content Refiner should:
            1. Make MAJOR, visible changes to the content structure and depth
            2. Double the information density with specific, actionable advice
            3. Expand total content length to at least 3000 words
            4. Add multiple detailed examples, case studies or scenarios
            5. Incorporate additional research and statistics from web searches
            6. Ensure content is suitable for ALL publishing platforms (not just LinkedIn)
            7. Transform any general advice into detailed, step-by-step guidance
            8. Add entirely new sections covering aspects not previously discussed
            9. Rewrite sections that didn't meet expectations completely
            
            The refined version should be dramatically improved and noticeably different from the previous draft.
            """
            
            blog_team.print_response(re_refinement_prompt, stream=True)
    
    # Step 5: SEO Optimization (after content refinement is approved)
    print("\n== SEO OPTIMIZATION ==")
    print("Optimizing content for search engines and readability...")
    
    blog_team.print_response(seo_optimization_prompt, stream=True)
    
    # SEO Optimization Review Loop
    seo_approved = False
    while not seo_approved:
        print("\n== SEO OPTIMIZATION REVIEW ==")
        seo_feedback = input("Is this SEO optimization good? (yes/no): ")
        
        if seo_feedback.lower() in ["yes", "y"]:
            seo_approved = True
        else:
            seo_user_feedback = input("What specific feedback do you have for improving the SEO optimization? ")
            context.store("seo_user_feedback", seo_user_feedback)
            
            print("\n== OPTIMIZING SEO AGAIN ==")
            print("Re-optimizing content based on your feedback...")
            
            re_seo_prompt = f"""
            Task: Further optimize the blog post SEO based on user feedback.
            
            User Feedback on SEO: {seo_user_feedback}
            
            The SEO Optimizer should address all feedback points while maintaining the content quality.
            Focus specifically on the SEO aspects mentioned in the feedback while ensuring the overall content
            remains natural and engaging for readers.
            
            Provide a fully optimized version that addresses all SEO feedback points.
            """
            
            blog_team.print_response(re_seo_prompt, stream=True)
    
    # Step 6: Final Polishing (after SEO optimization is approved)
    print("\n== FINAL POLISHING ==")
    print("Adding final touches to perfect the blog post...")
    
    polishing_prompt = f"""
    Task: Perfect the blog post with final professional touches for maximum impact across multiple publishing platforms.

    This task should be handled by the Final Editor, who should provide the following structured refinements:

    1. HEADLINE FINALIZATION:
       * Review and finalize the perfect headline
       * Ensure it balances SEO, engagement, and professionalism
       * Verify it accurately represents the content value
       * Confirm it appeals specifically to the target audience
       * Provide the final optimized headline with explanation

    2. OPENING OPTIMIZATION:
       * Fine-tune the crucial first 2-3 sentences
       * Ensure immediate value proposition is clear
       * Verify hook effectiveness and engagement potential
       * Perfect the transition into full content
       * Provide the final optimized opening

    3. STRUCTURAL PERFECTION:
       * Verify logical flow and progression of ideas
       * Ensure ideal paragraph length (3-4 sentences max)
       * Perfect the H2/H3 hierarchy and formatting
       * Confirm proper use of lists, bullets, and formatting elements
       * Address any remaining organization issues

    4. LANGUAGE PRECISION:
       * Eliminate any grammar or punctuation errors
       * Replace weak word choices with powerful alternatives
       * Remove any jargon not appropriate for audience
       * Ensure consistent tone and voice throughout
       * Perfect sentence structure variety and flow

    5. ENGAGEMENT MAXIMIZATION:
       * Strengthen calls-to-action throughout
       * Perfect the conversation-starting questions
       * Enhance personal connection elements
       * Strengthen authority positioning
       * Optimize shareability factors

    6. CROSS-PLATFORM FORMATTING:
       * Provide optimal formatting recommendations for WordPress
       * Include Medium-specific formatting guidelines
       * Add social media excerpt recommendations
       * Include community forum posting adaptations
       * Ensure formatting is adaptable across platforms

    7. FINAL PROFESSIONAL TOUCHES:
       * Add brief author perspective if appropriate
       * Verify all sources and attributions
       * Perfect the concluding statement
       * Ensure content exceeds audience expectations
       * Confirm unique value beyond the original video

    8. PUBLICATION PREPARATION:
       * Provide a final SEO title
       * Create a perfect meta description
       * Suggest featured image concept or description
       * Recommend categories and tags for various platforms
       * Include special formatting instructions for each platform

    The final post should be absolutely publication-ready, reflecting the highest standards of professional content while maximizing engagement, shareability, and SEO value across multiple publishing platforms.
    """
    
    blog_team.print_response(polishing_prompt, stream=True)
    
    # Final Approval Loop - ensure user is completely satisfied
    final_approved = False
    while not final_approved:
        print("\n== FINAL APPROVAL ==")
        final_approval = input("Is this blog post perfectly ready to publish  (yes/no): ")
        
        if final_approval.lower() in ["yes", "y"]:
            final_approved = True
        else:
            last_feedback = input("What final changes are needed before publishing? ")
            
            print("\n== MAKING FINAL CHANGES ==")
            print("Implementing your final changes...")
            
            final_changes_prompt = f"""
            Task: Make these final changes to the blog post before publication:
            
            User Feedback: {last_feedback}
            
            Create the absolutely final version that is ready for LinkedIn publishing and other platforms like facebook and instagram and twitter and medium and wordpress.
            Address each specific point of feedback while maintaining the overall quality,
            SEO optimization, and professional polish of the content.
            
            Provide the complete, publication-ready version with all requested changes implemented.
            """
            
            blog_team.print_response(final_changes_prompt, stream=True)
    
    # Step 8: Final Blog Post Output
    print("\n== FINAL BLOG POST ==")
    print("Preparing your blog post ..........")
    
    final_format_prompt = """
    Task: Deliver the complete, publication-ready blog post in perfect format for multiple publishing platforms.

    Provide the following structured output:

    1. PUBLICATION-READY BLOG POST:
       * Include the finalized headline (properly formatted)
       * Present the complete post with all formatting, spacing, and structure
       * Incorporate all optimized sections, transitions, and enhancements
       * Include the final call-to-action and engagement elements
       * Add the selected tags and categories in proper format

    2. SEO METADATA:
       * SEO Title: [Optimized title for search visibility]
       * Meta Description: [Compelling 150-155 character description]
       * Featured Image Suggestion: [Brief description of ideal image]
       * Primary Keyword: [Primary keyword for tracking]
       * Secondary Keywords: [3-5 secondary keywords]

    3. PLATFORM-SPECIFIC FORMATTING:
       * WordPress Version: [Any special WordPress formatting considerations]
       * Medium Version: [Any Medium-specific adaptations]
       * Community Forum Version: [How to adapt for forum posting]
       * Social Media Excerpts: [3-5 key excerpts for social sharing]

    4. PUBLICATION NOTES:
       * Optimal posting recommendations
       * Any special formatting instructions
       * Engagement monitoring suggestions
       * Follow-up content ideas

    Format the blog post with proper spacing, formatting, headings, and visual structure that can be easily adapted for multiple publishing platforms.
    """
    
    print("\nYour blog post is ready to publish:")
    print("=" * 80)
    blog_team.print_response(final_format_prompt, stream=True)
    print("=" * 80)
    
    # Offer to save the blog post to a file
    save_option = input("\nWould you like to save this blog post to a file? (yes/no): ")
    if save_option.lower() in ["yes", "y"]:
        file_name = input("Enter file name (default: multi_platform_blog_post.md): ") or "multi_platform_blog_post.md"
        
        # Capture the blog post content
        blog_content = capture_output(blog_team.print_response, final_format_prompt, stream=False)
        
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(blog_content)
        
        print(f"\nBlog post saved to {file_name}")
    
    print("\nWorkflow completed successfully!")

if __name__ == "__main__":
    main()


"""
This code creates a YouTube to  blog post workflow with the following steps:
-> Video Input: Takes user input for YouTube URL and target audience
-> Video Content Extraction: Extracts key information from the YouTube video
-> Keyword Research: Identifies valuable keywords for SEO optimization
-> Initial Content Creation: Drafts a blog post based on video content and keywords
-> Content Review: Gets user feedback and either proceeds or regenerates content
-> Content Refinement: Adds detailed information and improves overall quality
-> SEO Optimization: Enhances the content for search visibility
-> Final Polishing: Perfects the post for LinkedIn and other platforms like facebook and instagram and twitter and medium and wordpress platform requirements
-> Final Approval: Ensures user is completely satisfied with the content
-> Final Blog Post Output: Prepares and optionally saves the finalized blog post

The workflow uses specialized agents with different strengths:
- GPT-4o for extraction and creative writing (excels at long-form content)
- GPT-4o for keyword research and SEO (excels at technical SEO)
- A team coordination approach with agentic context for seamless collaboration
"""
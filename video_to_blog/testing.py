import os
import argparse
import logging
import json
import re
from typing import Dict, Any, Optional, List
from datetime import datetime
import sys
# Third-party imports
import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import requests
from yt_dlp import YoutubeDL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class LLMProvider:
    """Base class for various LLM providers"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, max_tokens: int = 4000):
        """
        Initialize the LLM provider.
        
        Args:
            model_name: Name of the model to use
            api_key: API key for the model (if required)
            max_tokens: Maximum tokens for output
        """
        self.model_name = model_name
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.llm = None
        
    def get_llm(self):
        """Get the initialized LLM instance"""
        if self.llm is None:
            self._initialize_llm()
        return self.llm
    
    def _initialize_llm(self):
        """Initialize the LLM (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def is_available(self) -> bool:
        """Check if the LLM is available"""
        try:
            self.get_llm()
            return True
        except Exception as e:
            logger.warning(f"LLM {self.model_name} is not available: {str(e)}")
            return False

class ClaudeProvider(LLMProvider):
    """Claude AI LLM provider"""
    
    def _initialize_llm(self):
        """Initialize Claude LLM"""
        if not self.api_key:
            self.api_key = os.getenv("CLAUDE_API_KEY")
            
        if not self.api_key:
            raise ValueError("Claude API key is required")
            
        self.llm = ChatAnthropic(
            model=self.model_name,
            anthropic_api_key=self.api_key,
            temperature=0.7,
            max_tokens=self.max_tokens
        )
        logger.info(f"Claude LLM initialized: {self.model_name}")

class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider"""
    
    def _initialize_llm(self):
        """Initialize OpenAI LLM"""
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
            
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
            
        self.llm = ChatOpenAI(
            model=self.model_name,
            openai_api_key=self.api_key,
            temperature=0.7,
            max_tokens=self.max_tokens
        )
        logger.info(f"OpenAI LLM initialized: {self.model_name}")

class OllamaProvider(LLMProvider):
    """Local Ollama LLM provider"""
    
    def __init__(self, model_name: str = "llama3", base_url: str = "http://localhost:11434", max_tokens: int = 4000):
        """
        Initialize the Ollama LLM provider.
        
        Args:
            model_name: Name of the model to use
            base_url: Base URL for the Ollama server
            max_tokens: Maximum tokens for output
        """
        super().__init__(model_name=model_name, max_tokens=max_tokens)
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
    def _initialize_llm(self):
        """Initialize Ollama LLM"""
        try:
            self.llm = Ollama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=0.7,
                num_predict=self.max_tokens
            )
            logger.info(f"Ollama LLM initialized: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {str(e)}")
            raise ValueError(f"Ollama initialization failed: {str(e)}")

class ImageSearcher:
    """Search and fetch relevant images for blog content"""
    
    def __init__(self, serp_api_key: Optional[str] = None, unsplash_api_key: Optional[str] = None):
        self.serp_api_key = serp_api_key or os.getenv("SERP_API_KEY")
        self.unsplash_api_key = unsplash_api_key or os.getenv("UNSPLASH_API_KEY")
        
    def search_images(self, query: str, num_images: int = 3) -> List[Dict[str, str]]:
        """Search for relevant images using multiple methods"""
        # Try all available methods
        methods = [
            self._search_with_serpapi,
            self._search_with_unsplash,
            self._search_with_pexels,
            self._get_placeholder_images
        ]
        
        for method in methods:
            try:
                images = method(query, num_images)
                if images:
                    return images
            except Exception as e:
                logger.warning(f"Image search method failed: {str(e)}")
                continue
        
        # If all methods fail, return placeholder
        return self._get_placeholder_images(query, num_images)
    
    def _search_with_serpapi(self, query: str, num_images: int) -> List[Dict[str, str]]:
        """Search images using SerpAPI"""
        if not self.serp_api_key:
            return []
        
        try:
            # Import here to avoid dependency issues
            from serpapi import GoogleSearch
            
            params = {
                "q": query,
                "tbm": "isch",
                "num": num_images,
                "api_key": self.serp_api_key
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            images = []
            for img in results.get("images_results", [])[:num_images]:
                images.append({
                    "url": img.get("original"),
                    "title": img.get("title", query),
                    "source": img.get("source", "Google Images")
                })
            
            return images
        except Exception as e:
            logger.error(f"SerpAPI search error: {str(e)}")
            return []
    
    def _search_with_unsplash(self, query: str, num_images: int) -> List[Dict[str, str]]:
        """Search images using Unsplash API"""
        if not self.unsplash_api_key:
            return []
        
        try:
            url = "https://api.unsplash.com/search/photos"
            headers = {"Authorization": f"Client-ID {self.unsplash_api_key}"}
            params = {"query": query, "per_page": num_images}
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            images = []
            for result in data.get("results", [])[:num_images]:
                images.append({
                    "url": result["urls"]["regular"],
                    "title": result.get("description", query) or result.get("alt_description", query) or query,
                    "source": f"Unsplash - {result['user']['name']}"
                })
            
            return images
        except Exception as e:
            logger.error(f"Unsplash search error: {str(e)}")
            return []
    
    def _search_with_pexels(self, query: str, num_images: int) -> List[Dict[str, str]]:
        """Search images using Pexels API"""
        pexels_api_key = ""
        if not pexels_api_key:
            return []
        
        try:
            url = f"https://api.pexels.com/v1/search?query={query}&per_page={num_images}"
            headers = {"Authorization": pexels_api_key}
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            images = []
            for photo in data.get("photos", [])[:num_images]:
                images.append({
                    "url": photo["src"]["large"],
                    "title": query,
                    "source": f"Pexels - {photo['photographer']}"
                })
            
            return images
        except Exception as e:
            logger.error(f"Pexels search error: {str(e)}")
            return []
    
    def _get_placeholder_images(self, query: str, num_images: int) -> List[Dict[str, str]]:
        """Return placeholder images when all other methods fail"""
        return [{
            "url": f"https://placehold.co/800x400/3498db/FFFFFF/png?text={query.replace(' ', '+')}",
            "title": query,
            "source": "Placeholder Image"
        }] * num_images

class YouTubeBlogGenerator:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
            
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            openai_api_key=self.api_key,
            temperature=0.7
        )
        
        self.image_searcher = ImageSearcher()
    
    def extract_video_content(self, url: str) -> Dict[str, Any]:
        """Enhanced video content extraction with multiple fallback methods"""
        logger.info(f"Extracting content from YouTube video: {url}")
        
        # Method 1: Try youtube-transcript-api
        try:
            # Extract video ID
            video_id = None
            if "youtube.com/watch?v=" in url:
                video_id = url.split("v=")[1].split("&")[0]
            elif "youtu.be/" in url:
                video_id = url.split("youtu.be/")[1].split("?")[0]
            else:
                raise ValueError("Invalid YouTube URL")
            
            # Get transcript
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US'])
            transcript_text = ' '.join([entry['text'] for entry in transcript_list])
            
            # Get metadata using oEmbed
            oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
            response = requests.get(oembed_url)
            
            if response.status_code == 200:
                metadata = response.json()
                return {
                    "title": metadata.get("title", "Untitled Video"),
                    "author": metadata.get("author_name", "Unknown Author"),
                    "description": metadata.get("description", ""),
                    "publish_date": datetime.now().strftime("%Y-%m-%d"),
                    "transcript": transcript_text,
                    "url": url
                }
        except Exception as e:
            logger.warning(f"Method 1 failed: {str(e)}")

        # Method 2: Try yt-dlp
        try:
            with YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                
                # Try to get transcript from subtitles
                transcript = ""
                if 'subtitles' in info and 'en' in info['subtitles']:
                    for sub in info['subtitles']['en']:
                        if sub['ext'] == 'vtt':
                            sub_response = requests.get(sub['url'])
                            lines = sub_response.text.splitlines()
                            for line in lines:
                                if '-->' not in line and line.strip():
                                    transcript += line.strip() + " "
                
                return {
                    "title": info.get('title', 'Untitled Video'),
                    "author": info.get('uploader', 'Unknown Author'),
                    "description": info.get('description', ''),
                    "publish_date": datetime.fromtimestamp(info.get('upload_date_timestamp', 0)).strftime("%Y-%m-%d"),
                    "transcript": transcript,
                    "url": url
                }
        except Exception as e:
            logger.warning(f"Method 2 failed: {str(e)}")

        # Method 3: Try original YoutubeLoader method
        try:
            loader = YoutubeLoader.from_youtube_url(
                url,
                add_video_info=True,
                language=["en", "en-US"]
            )
            video_data = loader.load()
            
            if video_data:
                return {
                    "title": video_data[0].metadata.get("title", "Untitled Video"),
                    "author": video_data[0].metadata.get("author", "Unknown Author"),
                    "description": video_data[0].metadata.get("description", ""),
                    "publish_date": video_data[0].metadata.get("publish_date", ""),
                    "transcript": video_data[0].page_content,
                    "url": url
                }
        except Exception as e:
            logger.warning(f"Method 3 failed: {str(e)}")

        raise ValueError("Failed to extract video content using all available methods")

    def generate_blog_post(self, video_info: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate a comprehensive blog post inspired by video content but expanded with additional insights.
        """
        logger.info("Generating blog post")
        
        try:
            # Create summary if not already available
            summary = self.summarize_transcript(video_info["transcript"])
            
            # Extract blog style and length preferences
            blog_style = analysis.get("style", "Professional")
            blog_length = analysis.get("length", "Comprehensive (4000+ words)")
            
            # Determine word count based on selected length
            min_word_count = 2000
            if "Detailed" in blog_length:
                min_word_count = 3000
            elif "Comprehensive" in blog_length:
                min_word_count = 4000
            
            # Extract related topics from analysis if available
            related_topics = []
            if "related_topics" in analysis:
                related_topics = analysis["related_topics"]
            elif "industry_trends" in analysis:
                related_topics = analysis["industry_trends"]
            
            # First, generate the blog structure - get themes from the video but allow for expansion
            structure_prompt = f"""
            Analyze this YouTube video transcript and create a detailed blog structure that uses the video as a foundation
            but expands into a more comprehensive exploration of the topic.
            
            Video Title: {video_info["title"]}
            Video Author: {video_info["author"]}
            
            Transcript Summary:
            {summary}
            
            Related Topics to Consider:
            {", ".join(related_topics) if related_topics else "Explore broader industry context, applications, and future trends"}
            
            Create an outline for a {min_word_count}+ word blog post that:
            1. Uses the video's core concepts as a starting point, not a strict limitation
            2. Expands beyond the video to provide additional depth and perspectives
            3. Includes sections that explore practical applications, case studies, and industry context
            4. Considers different angles and viewpoints not necessarily covered in the video
            5. Addresses potential challenges, limitations, or controversies in the field
            6. Looks forward to future trends and developments
            
            FORMAT YOUR RESPONSE AS JSON:
            {{
                "title": "A compelling H1 title that captures the essence of the expanded topic",
                "meta_description": "A 150-160 character description of the blog content",
                "introduction": "2-3 paragraphs introduction that references the video but sets up a broader exploration",
                "video_summary": "A brief overview of the original video's key points (will be expanded)",
                "sections": [
                    {{
                        "heading": "Main section heading (H2)",
                        "approach": "How this section expands beyond the video",
                        "subsections": [
                            {{
                                "heading": "Subsection heading (H3)",
                                "key_points": ["Point 1", "Point 2", "Point 3"],
                                "exploration_angle": "How to expand this subsection with additional context/insights"
                            }},
                            ... more subsections
                        ]
                    }},
                    ... more sections
                ],
                "conclusion": "A brief outline for the conclusion that ties everything together",
                "image_queries": ["Specific image query 1", "Specific image query 2"]
            }}
            
            IMPORTANT GUIDELINES:
            - Make sure the blog maintains connection to the video's core topic but is NOT limited to only what's mentioned
            - Include at least 6-8 main sections with 2-3 subsections each
            - Plan for sections that go deeper than the video into technical details, practical applications, or broader context
            - Include specific sections for real-world examples, case studies, or expert opinions beyond just the video creator
            """
            
            # Get the blog structure
            structure_response = self.llm.invoke(
                [{"role": "system", "content": "You are an expert content strategist who creates detailed blog structures that expand upon source material to create comprehensive, valuable content."},
                 {"role": "user", "content": structure_prompt}],
                temperature=0.6
            )
            
            structure_text = structure_response.content if hasattr(structure_response, 'content') else str(structure_response)
            
            # Extract JSON structure
            try:
                # Clean up any text outside of the JSON
                json_match = re.search(r'```json\s*(.*?)\s*```', structure_text, re.DOTALL)
                if json_match:
                    structure_text = json_match.group(1)
                else:
                    # Try to find JSON object
                    json_start = structure_text.find('{')
                    json_end = structure_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        structure_text = structure_text[json_start:json_end]
            
                blog_structure = json.loads(structure_text)
            
            except Exception as e:
                logger.warning(f"Failed to parse blog structure as JSON: {str(e)}")
                # If we can't parse the JSON, create a fallback structure
                blog_structure = {
                    "title": video_info["title"],
                    "meta_description": f"A comprehensive guide to {video_info['title']} by {video_info['author']}",
                    "introduction": "This blog post explores the key concepts discussed in the video.",
                    "sections": [{"heading": "Key Concepts", "subsections": [{"heading": "Overview", "key_points": ["Point 1"]}]}],
                    "conclusion": "In conclusion, this video provides valuable insights into the topic.",
                    "image_queries": [f"{video_info['title']} diagram", f"{video_info['title']} concept"]
                }

            # Now, generate the full content for each section based on the structure
            html_content = []
            
            # Add title
            html_content.append(f"<h1>{blog_structure['title']}</h1>")
            
            # Add introduction - Fix the backslash issue by using a different approach
            introduction_text = blog_structure['introduction']
            introduction_paragraphs = introduction_text.split('\n\n')
            introduction_html = ""
            for paragraph in introduction_paragraphs:
                if paragraph.strip():
                    introduction_html += f"<p>{paragraph}</p>\n"
            
            if not introduction_html:
                introduction_html = f"<p>{introduction_text}</p>"
            
            html_content.append(introduction_html)
            
            # Create table of contents
            toc_html = '<div class="table-of-contents">\n<h3>Table of Contents</h3>\n<ul>\n'
            
            for i, section in enumerate(blog_structure["sections"]):
                section_id = f"section-{i+1}"
                toc_html += f'<li><a href="#{section_id}">{section["heading"]}</a></li>\n'
            
            toc_html += '</ul>\n</div>\n'
            html_content.append(toc_html)
            
            # Generate content for each section
            for i, section in enumerate(blog_structure["sections"]):
                section_id = f"section-{i+1}"
                
                # Prepare context for expansion beyond the video
                expansion_approach = section.get("approach", "Expand on these concepts with additional context and examples")
                
                # Generate section content - encourage going beyond just the video content
                section_prompt = f"""
                Create detailed, insightful content for this blog section that uses the YouTube video as a foundation 
                but expands significantly beyond it.
                
                Video: "{video_info['title']}" by {video_info['author']}
                
                Section: {section['heading']}
                Expansion Approach: {expansion_approach}
                
                Video Summary:
                {summary[:1000]}
                
                Write comprehensive content (800-1000 words) for this section that:
                
                1. STARTS with insights from the video but EXPANDS beyond them by:
                   - Adding more depth, context, and nuance
                   - Including perspectives not mentioned in the video
                   - Connecting to broader industry trends or applications
                   - Providing additional examples or case studies
                
                2. Balances technical depth with accessibility by:
                   - Explaining complex concepts clearly for beginners
                   - Including advanced insights for experts
                   - Using analogies and real-world examples
                   - Breaking down technical terminology
                
                3. Enhances engagement with:
                   - Thought-provoking questions
                   - Highlighted key insights
                   - Practical tips and actionable advice
                   - Contrasting viewpoints or approaches
                
                4. Format using these HTML elements:
                   - Start with <h2 id="{section_id}">{section['heading']}</h2>
                   - Include <h3> headings for subsections
                   - Use <blockquote> for notable quotes or insights
                   - Add <div class="tip-box"> with practical advice
                   - Include <div class="highlight"> for important concepts
                   - Add <!-- IMAGE: "detailed specific image description" --> at appropriate spots
                   - Use <ul> or <ol> for lists
                
                For each subsection:
                - Add the H3 heading
                - Write 3-4 paragraphs that explore the topic in depth
                - Include specific examples, data points, or expert perspectives
                - Connect to related concepts and broader implications
                
                IMPORTANT: While you should reference concepts from the video, your primary goal is to create content that stands on its own and provides significant additional value beyond what's in the video.
                """
                
                section_response = self.llm.invoke(
                    [{"role": "system", "content": "You are an expert content writer who creates in-depth, valuable content that expands significantly beyond source material while maintaining relevance. You excel at adding context, nuance, and insights that weren't in the original source."},
                     {"role": "user", "content": section_prompt}],
                    temperature=0.7
                )
                
                section_content = section_response.content if hasattr(section_response, 'content') else str(section_response)
                
                # Clean up any markdown formatting
                section_content = re.sub(r'```html\s*|\s*```', '', section_content)
                
                # Ensure the section has proper HTML structure
                if not section_content.strip().startswith('<h2'):
                    section_content = f'<h2 id="{section_id}">{section["heading"]}</h2>\n{section_content}'
                else:
                    section_content = re.sub(r'<h2[^>]*>', f'<h2 id="{section_id}">', section_content, 1)
                
                html_content.append(f'<section id="{section_id}">\n{section_content}\n</section>')
            
            # Generate conclusion that ties everything together
            conclusion_prompt = f"""
            Create an insightful conclusion (300-400 words) for this expanded blog post about "{blog_structure['title']}".
            
            Video that inspired this blog: "{video_info['title']}" by {video_info['author']}
            
            Key sections covered in the blog:
            {', '.join([section['heading'] for section in blog_structure['sections']])}
            
            Write a conclusion that:
            1. Synthesizes insights from both the video and our expanded exploration
            2. Highlights key takeaways that provide value beyond what's in the original video
            3. Acknowledges different perspectives and approaches in the field
            4. Suggests how readers can apply these insights in practical ways
            5. Looks forward to future developments or trends
            6. Ends with thought-provoking questions or a call-to-action that encourages engagement
            
            Format as clean HTML with proper paragraph tags, using a conversational but authoritative tone.
            Start with <h2 id="conclusion">Conclusion</h2>
            """
            
            conclusion_response = self.llm.invoke(
                [{"role": "system", "content": "You are an expert content writer who creates effective conclusions for blog posts."},
                 {"role": "user", "content": conclusion_prompt}],
                temperature=0.6
            )
            
            conclusion_content = conclusion_response.content if hasattr(conclusion_response, 'content') else str(conclusion_response)
            conclusion_content = re.sub(r'```html\s*|\s*```', '', conclusion_content)
            
            if not conclusion_content.strip().startswith('<h2'):
                conclusion_content = f'<h2 id="conclusion">Conclusion</h2>\n{conclusion_content}'
            
            html_content.append(f'<section id="conclusion">\n{conclusion_content}\n</section>')
            
            # Add author bio and video source section
            source_section = f"""
            <section id="video-source">
                <h2>Video Source</h2>
                <p>This blog post is based on the YouTube video <strong>"{video_info['title']}"</strong> by <strong>{video_info['author']}</strong>. 
                The content strictly represents what was discussed in the video. For more information, we recommend watching the 
                original video and subscribing to the creator's channel.</p>
            </section>
            """
            html_content.append(source_section)
            
            # Add source attribution with light connection to the video
            source_attribution = f"""
            <section id="attribution" class="attribution-section">
                <h2>Source and Further Reading</h2>
                <div class="source-info">
                    <p>This blog post was inspired by the YouTube video <strong>"{video_info['title']}"</strong> by <strong>{video_info['author']}</strong>. 
                    While we've expanded significantly on the concepts discussed, we recommend watching the 
                    <a href="#" target="_blank">original video</a> for additional insights.</p>
                    
                    <h3>Further Reading</h3>
                    <ul class="further-reading">
                        <li>Industry reports and whitepapers on {video_info['title'].split()[0]}</li>
                        <li>Academic research on the latest developments in this field</li>
                        <li>Case studies of practical applications</li>
                    </ul>
                </div>
            </section>
            """
            html_content.append(source_attribution)
            
            # Combine all content
            full_content = '\n'.join(html_content)
            
            # Extract image queries from the content
            image_markers = re.findall(r'<!-- IMAGE: "(.*?)" -->', full_content)
            
            # Ensure we have highly specific image queries related to the video
            improved_image_queries = []
            video_keywords = ' '.join(video_info['title'].split()[:3])  # Use first 3 words from title
            
            for marker in image_markers:
                # Make the query more specific by adding the video title keywords
                if video_keywords.lower() not in marker.lower():
                    improved_query = f"{video_keywords} {marker}"
                else:
                    improved_query = marker
                    
                # Add descriptive visual terms if not present
                if not any(term in improved_query.lower() for term in ['diagram', 'illustration', 'infographic', 'chart', 'example']):
                    improved_query += " diagram illustration"
                    
                improved_image_queries.append(improved_query)
            
            # Add more image queries if needed
            while len(improved_image_queries) < 5:
                section_index = len(improved_image_queries) % len(blog_structure["sections"])
                section = blog_structure["sections"][section_index]
                query = f"{video_info['title']} {section['heading']} diagram illustration"
                improved_image_queries.append(query)
            
            # Process images with more specific queries
            content_with_images = self._process_image_markers(full_content, improved_image_queries)
            
            # Create the final blog data
            blog_data = {
                "title": blog_structure["title"],
                "meta_description": blog_structure["meta_description"],
                "content": content_with_images
            }
            
            # Add schema markup
            blog_data["content"] = self.add_schema_markup(
                blog_data["content"],
                blog_data["title"],
                blog_data["meta_description"],
                video_info
            )
            
            # Format with HTML template
            html_content = self.generate_html(blog_data)
            
            logger.info("Blog post generation completed successfully")
            return {
                "title": blog_data["title"],
                "meta_description": blog_data["meta_description"],
                "content": html_content
            }
        
        except Exception as e:
            logger.error(f"Error generating blog post: {str(e)}")
            return {
                "title": video_info["title"],
                "meta_description": f"A guide to {video_info['title']} by {video_info['author']}",
                "content": f"<h1>{video_info['title']}</h1>\n\n<p>Error generating blog content: {str(e)}</p>"
            }

    def _process_image_markers(self, content: str, image_queries: List[str]) -> str:
        """
        Process image markers in content and replace with topic-relevant images.
        
        Args:
            content: HTML content with image markers
            image_queries: List of image search queries
            
        Returns:
            HTML content with actual images
        """
        logger.info(f"Processing {len(image_queries)} image queries")
        
        # Create image searcher if not already available
        if not hasattr(self, 'image_searcher'):
            self.image_searcher = ImageSearcher()
        
        # Process each query
        for i, query in enumerate(image_queries):
            try:
                # Make sure query is specific and relevant
                if len(query.split()) < 3:
                    # Too generic, make it more specific
                    query = f"detailed diagram of {query} concept"
                
                # Search for images
                image_results = self.image_searcher.search_images(query, num_images=1)
                    
                if image_results and len(image_results) > 0:
                    image = image_results[0]
                    
                    # Verify image URL
                    if not image['url'].startswith(('http://', 'https://')):
                        # Use placeholder instead
                        image['url'] = f"https://placehold.co/800x400/3498db/FFFFFF/png?text={query.replace(' ', '+')}"
                        image['source'] = "Placeholder Image"
                    
                    # Create HTML for the image
                    img_html = f"""
                    <figure class="blog-image">
                        <img src="{image['url']}" 
                            alt="{image['title'] or query}"
                            loading="lazy">
                        <figcaption>{image['title'] or query} 
                        <br><small>(Source: {image['source']})</small></figcaption>
                    </figure>
                    """
                    
                    # Find the appropriate marker to replace
                    marker_found = False
                    for marker in [f'<!-- IMAGE: "{query}" -->', '<!-- IMAGE PLACEHOLDER -->']:
                        if marker in content:
                            content = content.replace(marker, img_html, 1)
                            marker_found = True
                            break
                    
                    # If no marker found, try to find any image marker
                    if not marker_found:
                        marker_pattern = r'<!-- IMAGE: ".*?" -->'
                        match = re.search(marker_pattern, content)
                        if match:
                            content = content.replace(match.group(0), img_html, 1)
                        else:
                            # Add image at the end of an appropriate section
                            sections = re.findall(r'(<section id=".*?">.*?</section>)', content, re.DOTALL)
                            if i < len(sections):
                                section = sections[i]
                                section_with_image = section.replace('</section>', f'{img_html}</section>')
                                content = content.replace(section, section_with_image)
                else:
                    logger.warning(f"No images found for query: {query}")
                    
                    # Add a placeholder image with the query text
                    placeholder_html = f"""
                    <figure class="blog-image">
                        <img src="https://placehold.co/800x400/3498db/FFFFFF/png?text={query.replace(' ', '+')}" 
                            alt="{query}"
                            loading="lazy">
                        <figcaption>{query} <br><small>(Placeholder Image)</small></figcaption>
                    </figure>
                    """
                    
                    # Find any image marker to replace
                    marker_pattern = r'<!-- IMAGE: ".*?" -->'
                    match = re.search(marker_pattern, content)
                    if match:
                        content = content.replace(match.group(0), placeholder_html, 1)
                
            except Exception as e:
                logger.error(f"Error processing image query '{query}': {str(e)}")
        
        return content

    def add_schema_markup(self, content: str, title: str, meta_description: str, video_info: Dict[str, Any]) -> str:
        """
        Add schema.org markup for SEO with video information.
        
        Args:
            content: HTML content
            title: Blog title
            meta_description: Meta description
            video_info: Dictionary with video information
            
        Returns:
            Enhanced HTML with schema markup
        """
        # Generate a more detailed schema that includes video data
        schema = {
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": title,
            "description": meta_description,
            "image": video_info.get("thumbnail_url", ""),
            "datePublished": datetime.now().strftime("%Y-%m-%d"),
            "author": {
                "@type": "Person",
                "name": "YouTube Blog Generator"
            },
            "publisher": {
                "@type": "Organization",
                "name": "YouTube Blog Generator",
                "logo": {
                    "@type": "ImageObject",
                    "url": "https://example.com/logo.png"  # Replace with actual logo
                }
            },
            "mainEntityOfPage": {
                "@type": "WebPage",
                "@id": "https://example.com/blog/" + datetime.now().strftime("%Y%m%d")  # Replace with actual URL
            },
            "about": {
                "@type": "Thing",
                "name": video_info.get("title", "")
            },
            "video": {
                "@type": "VideoObject",
                "name": video_info.get("title", ""),
                "description": video_info.get("description", ""),
                "thumbnailUrl": video_info.get("thumbnail_url", ""),
                "uploadDate": video_info.get("publish_date", datetime.now().strftime("%Y-%m-%d")),
                "publisher": {
                    "@type": "Organization",
                    "name": video_info.get("author", "")
                }
            }
        }
        
        schema_script = f'<script type="application/ld+json">{json.dumps(schema)}</script>\n\n'
        
        # Add table of contents if not already present
        if '<div class="table-of-contents">' not in content:
            toc = '<div class="table-of-contents">\n<h3>Table of Contents</h3>\n<ul>\n'
            
            headings = re.findall(r'<h2[^>]*>(.*?)</h2>', content)
            for i, heading in enumerate(headings):
                anchor_id = f"section-{i+1}"
                content = content.replace(
                    f'<h2>{heading}</h2>',
                    f'<h2 id="{anchor_id}">{heading}</h2>',
                    1
                )
                toc += f'<li><a href="#{anchor_id}">{heading}</a></li>\n'
            
            toc += '</ul>\n</div>\n\n'
            
            first_para_end = content.find('</p>')
            first_h1_end = content.find('</h1>')
            
            insert_pos = max(first_para_end, first_h1_end)
            if insert_pos > 0:
                insert_pos += 4
                enhanced_content = content[:insert_pos] + '\n\n' + toc + content[insert_pos:]
            else:
                enhanced_content = toc + content
        else:
            enhanced_content = content
        
        # Ensure all sections are properly wrapped in section tags
        if '<section' not in enhanced_content:
            # Find all h2 headings
            h2_pattern = r'<h2[^>]*>.*?</h2>'
            h2_matches = list(re.finditer(h2_pattern, enhanced_content))
            
            # Wrap content between h2 tags in section elements
            for i in range(len(h2_matches)):
                start = h2_matches[i].start()
                end = h2_matches[i+1].start() if i < len(h2_matches) - 1 else len(enhanced_content)
                section_id = f"section-{i+1}"
                
                section_content = enhanced_content[start:end]
                section_wrapped = f'<section id="{section_id}">\n{section_content}\n</section>'
                
                enhanced_content = enhanced_content[:start] + section_wrapped + enhanced_content[end:]
        
        return schema_script + enhanced_content

    def generate_html(self, blog_data: Dict[str, str]) -> str:
        """
        Generate HTML from blog data with professional styling.
        
        Args:
            blog_data: Dictionary with blog data
            
        Returns:
            HTML content
        """
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <meta name="description" content="{meta_description}">
            <title>{title}</title>
            <style>
                :root {{
                    --primary-color: #2c3e50;
                    --secondary-color: #3498db;
                    --accent-color: #e74c3c;
                    --light-bg: #f8f9fa;
                    --dark-bg: #2c3e50;
                    --text-color: #333;
                    --light-text: #f8f9fa;
                    --border-radius: 5px;
                    --box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                
                * {{
                    box-sizing: border-box;
                    margin: 0;
                    padding: 0;
                }}
                
                body {{
                    font-family: 'Segoe UI', Roboto, -apple-system, BlinkMacSystemFont, sans-serif;
                    line-height: 1.7;
                    color: var(--text-color);
                    background-color: #fff;
                    max-width: 850px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                
                /* Typography */
                h1, h2, h3, h4, h5, h6 {{
                    margin-top: 1.5em;
                    margin-bottom: 0.8em;
                    line-height: 1.3;
                    color: var(--primary-color);
                    font-weight: 700;
                }}
                
                h1 {{
                    font-size: 2.5rem;
                    text-align: center;
                    margin-top: 1em;
                    color: var(--primary-color);
                    border-bottom: 2px solid var(--secondary-color);
                    padding-bottom: 0.5em;
                }}
                
                h2 {{
                    font-size: 1.8rem;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 0.3em;
                    margin-top: 2em;
                }}
                
                h3 {{
                    font-size: 1.4rem;
                    color: var(--secondary-color);
                }}
                
                p {{
                    margin-bottom: 1.5em;
                    font-size: 1.1rem;
                }}
                
                a {{
                    color: var(--secondary-color);
                    text-decoration: none;
                    transition: color 0.2s;
                    border-bottom: 1px dotted var(--secondary-color);
                }}
                
                a:hover {{
                    color: var(--accent-color);
                    border-bottom: 1px solid var(--accent-color);
                }}
                
                /* Blog Structure */
                .table-of-contents {{
                    background-color: var(--light-bg);
                    padding: 1.5em;
                    border-radius: var(--border-radius);
                    margin: 2em 0;
                    box-shadow: var(--box-shadow);
                }}
                
                .table-of-contents h3 {{
                    margin-top: 0;
                    margin-bottom: 1em;
                    text-align: center;
                }}
                
                .table-of-contents ul {{
                    list-style-type: none;
                    padding-left: 0;
                }}
                
                .table-of-contents ul li {{
                    margin-bottom: 0.7em;
                    padding-left: 1.5em;
                    position: relative;
                }}
                
                .table-of-contents ul li::before {{
                    content: "→";
                    position: absolute;
                    left: 0;
                    color: var(--secondary-color);
                }}
                
                /* Content Elements */
                blockquote {{
                    border-left: 4px solid var(--secondary-color);
                    padding: 1em 1.5em;
                    margin: 1.5em 0;
                    background-color: rgba(52, 152, 219, 0.1);
                    border-radius: 0 var(--border-radius) var(--border-radius) 0;
                    font-style: italic;
                }}
                
                blockquote p:last-child {{
                    margin-bottom: 0;
                }}
                
                .tip-box {{
                    background-color: rgba(46, 204, 113, 0.1);
                    border-left: 4px solid #2ecc71;
                    padding: 1.5em;
                    margin: 1.5em 0;
                    border-radius: 0 var(--border-radius) var(--border-radius) 0;
                }}
                
                .tip-box::before {{
                    content: "💡 Tip";
                    display: block;
                    font-weight: bold;
                    margin-bottom: 0.5em;
                    color: #27ae60;
                }}
                
                .highlight {{
                    background-color: rgba(241, 196, 15, 0.1);
                    border-left: 4px solid #f1c40f;
                    padding: 1.5em;
                    margin: 1.5em 0;
                    border-radius: 0 var(--border-radius) var(--border-radius) 0;
                }}
                
                .highlight::before {{
                    content: "🔑 Key Point";
                    display: block;
                    font-weight: bold;
                    margin-bottom: 0.5em;
                    color: #f39c12;
                }}
                
                /* Code blocks */
                code {{
                    background-color: #f5f5f5;
                    padding: 0.2em 0.4em;
                    border-radius: 3px;
                    font-family: Consolas, Monaco, 'Andale Mono', monospace;
                    font-size: 0.9em;
                }}
                
                pre {{
                    background-color: #f5f5f5;
                    padding: 1em;
                    border-radius: var(--border-radius);
                    overflow-x: auto;
                    margin: 1.5em 0;
                }}
                
                pre code {{
                    background-color: transparent;
                    padding: 0;
                }}
                
                /* Images */
                .blog-image {{
                    margin: 2em 0;
                    text-align: center;
                }}
                
                .blog-image img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: var(--border-radius);
                    box-shadow: var(--box-shadow);
                }}
                
                figcaption {{
                    margin-top: 0.8em;
                    font-size: 0.9em;
                    color: #777;
                }}
                
                /* Lists */
                ul, ol {{
                    margin: 1em 0 1.5em 2em;
                }}
                
                li {{
                    margin-bottom: 0.5em;
                }}
                
                /* Table */
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 2em 0;
                }}
                
                th, td {{
                    padding: 0.75em;
                    border: 1px solid #ddd;
                }}
            </style>
        </head>
        <body>
            <article itemscope itemtype="https://schema.org/Article">
                <meta itemprop="headline" content="{title}">
                <meta itemprop="description" content="{meta_description}">
                <meta itemprop="datePublished" content="{date}">
                
                {content}
                
                <footer>
                    <p>This blog was generated based on a YouTube video. All content is directly derived from the video.</p>
                    <p>Generated on {date}</p>
                </footer>
            </article>
        </body>
        </html>
        """
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        html_content = html_template.format(
            title=blog_data['title'],
            meta_description=blog_data['meta_description'],
            content=blog_data['content'],
            date=today
        )
        
        return html_content

    def summarize_transcript(self, transcript: str) -> str:
        """
        Generate a factual summary of the video transcript.
        
        Args:
            transcript: Video transcript text
            
        Returns:
            Summarized transcript
        """
        logger.info("Summarizing video transcript")
        
        try:
            # Split transcript into chunks for processing
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=4000,
                chunk_overlap=400
            )
            texts = text_splitter.split_text(transcript)
            
            # Use direct summarization approach to maintain factuality
            summary_messages = [
                {"role": "system", "content": """You are an expert at creating factual, comprehensive summaries of video transcripts.
                Your job is to extract the key points, main arguments, and important details from the transcript WITHOUT adding any 
                information not present in the text. Focus ONLY on what is explicitly stated, not on interpretations or additions.
                NEVER include information not directly stated in the transcript."""},
                {"role": "user", "content": f"""Create a detailed, factual summary of this video transcript. 
                Focus only on what is explicitly stated in the transcript. Include:
                1. Main topics and concepts explained
                2. Key definitions and explanations provided
                3. Examples used by the speaker
                4. Conclusions or calls to action mentioned
                
                Transcript:
                {transcript[:15000]}"""}
            ]
            
            summary_response = self.llm.invoke(summary_messages, temperature=0.3)
            summary = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)
            
            logger.info("Transcript summarization completed")
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing transcript: {str(e)}")
            # Return a portion of the transcript if summarization fails
            return transcript[:2000] + "..."

    def extract_blog_content(self, content: str) -> Dict[str, str]:
        """
        Extract blog content from LLM response when JSON parsing fails.
        
        Args:
            content: Raw response content
            
        Returns:
            Dictionary with blog data
        """
        try:
            title_match = re.search(r'title["\s:]+([^"}\n]+)', content)
            title = title_match.group(1) if title_match else "Video Analysis"
            
            meta_match = re.search(r'meta_description["\s:]+([^"}\n]+)', content)
            meta_description = meta_match.group(1) if meta_match else "Analysis of video content"
            
            content_text = ""
            in_content = False
            for line in content.split('\n'):
                if '<h1>' in line or '<p>' in line:
                    in_content = True
                if in_content:
                    content_text += line + '\n'
                if '</body>' in line or '"}' in line:
                    in_content = False
            
            if not content_text:
                content_start = content.find('"content":')
                if content_start > 0:
                    content_text = content[content_start+10:]
                    content_text = content_text.strip('"}').strip()
            
            return {
                "title": title.strip('"\' '),
                "meta_description": meta_description.strip('"\' '),
                "content": content_text.strip()
            }
        except Exception as e:
            logger.error(f"Error extracting blog content: {str(e)}")
            return {
                "title": "Video Analysis",
                "meta_description": "Analysis of video content",
                "content": content
            }

def main():
    """Streamlit app for blog generation"""
    st.set_page_config(
        page_title="AI Blog Generator",
        page_icon="",
        layout="wide"
    )
    
    st.title("📝 AI Blog Generator")
    st.markdown("""
    Transform YouTube videos into high-quality, engaging blog posts using GPT-4.
    Our AI analyzes video content and creates comprehensive blog posts with relevant images.
    """)
    
    # API key input
    with st.sidebar:
        st.header("🔑 API Keys")
        
        openai_api_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            help="Enter your OpenAI API key"
        )
        
        serp_api_key = st.text_input(
            "SerpAPI Key (Optional)",
            type="password",
            help="Enter your SerpAPI key for image search"
        )
    
    # Main interface
    url = st.text_input(
        "YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )
    
    # New options
    blog_style = st.selectbox(
        "Blog Style",
        ["Professional", "Technical", "Educational"],
        help="Select the writing style for your blog"
    )

    blog_length = st.selectbox(
        "Content Length",
        [
            "Standard (2000-3000 words)",
            "Detailed (3000-4000 words)",
            "Comprehensive (4000+ words)"
        ],
        help="Select the desired length for your blog post"
    )
    
    if st.button("Generate Blog", type="primary"):
        if not openai_api_key:
            st.error("Please enter your OpenAI API key")
            return
            
        if not url:
            st.error("Please enter a YouTube URL")
            return
        
        try:
            with st.spinner("Generating your blog post..."):
                # Initialize generator
                generator = YouTubeBlogGenerator(api_key=openai_api_key)
                
                # Extract video content
                video_info = generator.extract_video_content(url)
                
                # Generate blog
                blog = generator.generate_blog_post(
                    video_info, 
                    {
                        "style": blog_style,
                        "length": blog_length
                    }
                )
                
                # Display result
                st.success("Blog generated successfully!")
                
                # Show blog preview
                st.subheader("Blog Preview")
                st.components.v1.html(blog["content"], height=800, scrolling=True)
                
                # Download options
                st.download_button(
                    "Download HTML",
                    blog["content"],
                    file_name=f"blog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            logger.error(f"Error in main app: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
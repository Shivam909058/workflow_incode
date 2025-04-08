# so we are creating a tool to convert video to blog post
# but first i am creating a tool to check score of a blog generated by AI , and then we will intregate this tool to the main tool which will be used to convert video to blog post

import json
import re
import requests
import textstat
from openai import OpenAI
import nltk
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from bs4.element import Comment, Doctype
import time
nltk.download("stopwords")
from nltk.corpus import stopwords
openai = OpenAI(api_key="")


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_blog_text(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        print(f"Attempting to fetch URL: {url}")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        html_content = response.text
        
        print(f"Successfully fetched HTML content ({len(html_content)} bytes)")
        
        if "videotoblog.ai" in url:
            print("Detected videotoblog.ai site, using specialized extraction...")
            json_data_match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html_content, re.DOTALL)
            json_data_match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html_content, re.DOTALL)
            if json_data_match:
                try:
                    json_data = json.loads(json_data_match.group(1))
                    if 'props' in json_data and 'pageProps' in json_data['props']:
                        page_props = json_data['props']['pageProps']
                        if 'post' in page_props and 'content' in page_props['post']:
                            content = page_props['post']['content']
                            if content.startswith('<'):
                                content = extract_text_from_html(content)
                            print(f"Successfully extracted content from JSON data: {len(content)} characters")
                            return content
                except Exception as json_err:
                    print(f"Error parsing JSON data: {json_err}")
        
        content = extract_text_from_html(html_content)
        
        if len(content) < 200:
            print("Content too short, trying alternative extraction methods...")
            
            soup = BeautifulSoup(html_content, "html.parser")
            body = soup.find('body')
            if body:
                texts = body.find_all(text=lambda text: isinstance(text, str) and text.strip())
                visible_texts = [t.strip() for t in texts if tag_visible(t.parent)]
                content = " ".join(t for t in visible_texts if t)
                content = clean_text(content)
                print(f"Method 1 extracted {len(content)} characters")
            
            if len(content) < 200:
                soup = BeautifulSoup(html_content, "html.parser")
                content_divs = soup.find_all('div', class_=lambda c: c and ('content' in c.lower() or 'article' in c.lower() or 'post' in c.lower()))
                if content_divs:
                    content = " ".join(div.get_text(separator=' ', strip=True) for div in content_divs)
                    content = clean_text(content)
                    print(f"Method 2 extracted {len(content)} characters")
            
            if len(content) < 200:
                soup = BeautifulSoup(html_content, "html.parser")
                elements_with_data = soup.find_all(attrs=lambda attrs: any(k.startswith('data-') for k in attrs.keys() if k))
                data_texts = []
                for elem in elements_with_data:
                    for attr_name, attr_value in elem.attrs.items():
                        if attr_name.startswith('data-') and isinstance(attr_value, str) and len(attr_value) > 100:
                            data_texts.append(attr_value)
                if data_texts:
                    content = " ".join(data_texts)
                    content = clean_text(content)
                    print(f"Method 3 extracted {len(content)} characters")
        
        print(f"Final extraction: {len(content)} characters")
        
        if len(content) < 200:
            print("Warning: Limited content extracted. This is likely a JavaScript-rendered site.")
            print("Consider using a headless browser for better extraction.")
            
            soup = BeautifulSoup(html_content, "html.parser")
            all_text_elements = soup.find_all(text=True)
            large_text_blocks = [t for t in all_text_elements if len(t.strip()) > 100]
            if large_text_blocks:
                content = " ".join(t.strip() for t in large_text_blocks)
                content = clean_text(content)
                print(f"Last resort method extracted {len(content)} characters")
        
        return content
    except Exception as e:
        print(f"Error extracting blog text from URL: {e}")
        return ""


def extract_text_from_html(html_content):
    """
    Common function to extract text from HTML content.
    Used by both URL and file extraction to ensure consistency.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.extract()
    
    content = ""
    
    main_content = soup.find('article') or soup.find('main') or soup.find('div', class_='content')
    
    if main_content:
        content = main_content.get_text(separator=' ', strip=True)
    else:
        paragraphs = soup.find_all("p")
        if paragraphs:
            content = " ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
        
        if not content or len(content) < 100:
            body = soup.find('body')
            if body:
                content = body.get_text(separator=' ', strip=True)
    
    content = clean_text(content)
    
    return content


def readability_score(text):
    try:
        return textstat.flesch_reading_ease(text)
    except Exception as e:
        print(f"Error computing readability score: {e}")
        
        return 0


def seo_score(text):
    words = text.split()
    word_count = len(words)
    
    if word_count < 300:
        score = 30  # Penalize short content
    elif 300 <= word_count <= 1500:
        score = 70
    else:
        score = 90  # Longer blogs tend to rank better

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_array = vectorizer.get_feature_names_out()
        keyword_density = len(feature_array) / word_count if word_count > 0 else 0
    except Exception as e:
        print(f"Error computing TF-IDF: {e}")
        keyword_density = 0

    if 0.01 < keyword_density < 0.05:
        score += 10
    else:
        score -= 10

    return min(max(score, 0), 100)


def call_openai_chat(prompt):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"}  # Request JSON format directly
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retrying
            else:
                return None


def parse_json_response(response_text):
    try:
        # Check if the response is wrapped in a code block
        code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if code_block_match:
            json_text = code_block_match.group(1)
            return json.loads(json_text)
        return json.loads(response_text)
    except Exception as e:
        print(f"JSON parsing error: {e}. Response was: {response_text}")
        # Try to extract any JSON-like structure from the text
        json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
        json_match = re.search(json_pattern, response_text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass
        return {}


def ai_detection(text):
    prompt = (
        "You are an expert in content analysis. Analyze the following text "
        "and determine if it is AI-generated or human-written. Provide your answer as a JSON object with two keys: "
        "'score' (an integer between 0 and 100, where 0 means entirely human and 100 means entirely AI) and "
        "'explanation' (a brief explanation of your assessment).\n\n"
        f"Text:\n{text}\n"
    )
    response_text = call_openai_chat(prompt)
    result = parse_json_response(response_text) if response_text else {}
    if "score" not in result:
        result["score"] = 50
        result["explanation"] = "Could not parse AI detection score."
    return result


def grammar_score(text):
    prompt = (
        "You are an expert in language quality assessment. Rate the grammar and clarity of the following text "
        "on a scale from 0 to 100 (where 100 indicates perfect grammar and clarity) and provide a brief explanation. "
        "Respond as a JSON object with keys 'score' and 'explanation'.\n\n"
        f"Text:\n{text}\n"
    )
    response_text = call_openai_chat(prompt)
    result = parse_json_response(response_text) if response_text else {}
    if "score" not in result:
        result["score"] = 50
        result["explanation"] = "Could not parse grammar score."
    return result


def plagiarism_check(text):
    # For now, this function returns a fixed value.
    return 0  # Modify this to integrate a real plagiarism checker.


def structure_score(text):
    headings = len(re.findall(r"<h[1-6]>", text))  # if HTML is provided
    bullet_points = len(re.findall(r"[\*\-]\s", text))  # markdown bullets
    paragraphs = text.count("\n\n")  # rough paragraph count in plain text

    # A simple heuristic: more structure elements yield a higher score.
    base_score = 50
    structure_bonus = min(headings * 5 + bullet_points * 3 + paragraphs * 2, 50)
    return min(base_score + structure_bonus, 100)


def analyze_blog(text):
    text = clean_text(text)
    word_count = len(text.split())
    
        
    content_metrics = {
        "Word Count": word_count,
        "Estimated Read Time": f"{round(word_count/200, 1)} minutes",  # Average reading speed
        "Paragraphs": len(text.split('\n\n')),
    }
    
    readability_metrics = {
        "Flesch Reading Ease": textstat.flesch_reading_ease(text),
        "Grade Level": textstat.coleman_liau_index(text),
        "Text Complexity": textstat.text_standard(text),
        "Sentence Count": textstat.sentence_count(text)
    }
    
    seo_metrics = {
        "Basic SEO Score": seo_score(text),
        "Keyword Density": analyze_keyword_density(text),
        "Meta Description Length": check_meta_description(text),
        "Header Structure": analyze_headers(text)
    }
    
    quality_metrics = {
        "Grammar Score": grammar_score(text),
        "AI Detection": ai_detection(text),
        "Engagement Score": calculate_engagement_score(text),
        "Content Depth": analyze_content_depth(text)
    }
    
    final_score = calculate_final_score(
        readability_metrics,
        seo_metrics,
        quality_metrics,
        content_metrics
    )
    
    return {
        "Content Metrics": content_metrics,
        "Readability Analysis": readability_metrics,
        "SEO Analysis": seo_metrics,
        "Quality Metrics": quality_metrics,
        "Final Score": final_score
    }


def analyze_keyword_density(text):
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10,  # Top 10 keywords
            ngram_range=(1, 2)  # Include bigrams
        )
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        
        keywords = {
            feature_names[i]: round(scores[i] * 100, 2)
            for i in scores.argsort()[-5:][::-1]  # Top 5 keywords
        }
        
        return keywords
    except Exception as e:
        print(f"Error in keyword analysis: {e}")
        return {}


def calculate_engagement_score(text):
    prompt = (
        "Analyze this text for reader engagement potential. Consider:\n"
        "1. Use of hooks and compelling openings\n"
        "2. Story elements and narrative flow\n"
        "3. Call-to-actions and reader interaction\n"
        "4. Use of examples and illustrations\n"
        "Provide a score (0-100) and brief explanation as JSON.\n\n"
        f"Text:\n{text[:1000]}..."  # First 1000 chars for analysis
    )
    response = call_openai_chat(prompt)
    result = parse_json_response(response) if response else {}
    return result


def analyze_content_depth(text):
    prompt = (
        "Analyze this content for depth and expertise. Consider:\n"
        "1. Topic coverage completeness\n"
        "2. Use of technical terms and explanations\n"
        "3. Presence of supporting evidence\n"
        "4. Balance of basic and advanced concepts\n"
        "Provide a score (0-100) and explanation as JSON.\n\n"
        f"Text:\n{text[:1500]}..."  # First 1500 chars for analysis
    )
    response = call_openai_chat(prompt)
    result = parse_json_response(response) if response else {}
    return result


def calculate_final_score(readability, seo, quality, content):
    try:
        score = 70
        
        if content["Word Count"] < 300:
            score -= 15
        elif content["Word Count"] > 1500:
            score += 10
        
        if 60 <= readability["Flesch Reading Ease"] <= 70:
            score += 10
        
        if isinstance(seo["Basic SEO Score"], (int, float)):
            score += (seo["Basic SEO Score"] - 50) * 0.2
        
        if isinstance(quality["Grammar Score"], dict) and "score" in quality["Grammar Score"]:
            score += (quality["Grammar Score"]["score"] - 50) * 0.3
        
        if isinstance(quality["AI Detection"], dict) and "score" in quality["AI Detection"]:
            if quality["AI Detection"]["score"] > 80:
                score -= 15
        
        return max(min(round(score, 2), 100), 0)  # Ensure score is between 0 and 100
    except Exception as e:
        print(f"Error calculating final score: {e}")
        return 50


def analyze_headers(text):
    markdown_headers = re.findall(r'^#{1,6}\s.+$', text, re.MULTILINE)
    html_headers = re.findall(r'<h[1-6]>.*?</h[1-6]>', text, re.IGNORECASE)
    
    return {
        "Header Count": len(markdown_headers) + len(html_headers),
        "Has H1": bool(re.search(r'^#\s|<h1>', text, re.MULTILINE|re.IGNORECASE)),
        "Header Hierarchy": "Good" if len(markdown_headers) + len(html_headers) >= 3 else "Needs Improvement"
    }


def check_meta_description(text):
    first_para = text.split('\n\n')[0] if text else ""
    return {
        "Length": len(first_para),
        "Quality": "Good" if 120 <= len(first_para) <= 160 else "Needs Adjustment"
    }


def read_blog_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Check if it's an HTML file
        if file_path.lower().endswith(('.html', '.htm')):
            print("Detected HTML file, parsing content...")
            content = extract_text_from_html(content)
            print(f"Extracted {len(content)} characters from HTML file")
        else:
            # For plain text files, just clean the text
            content = clean_text(content)
            
        return content
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, (Comment, Doctype)):
        return False
    return True


if __name__ == "__main__":
    print("Enter blog file path or URL:")
    source = input().strip()
    
    if not source:
        source = "https://www.videotoblog.ai/share/pM6KiPDrwgnWlps7fg2c"
        print(f"Using default URL: {source}")
    
    if source.startswith(('http://', 'https://')):
        print(f"Fetching content from URL: {source}")
        sample_blog = extract_blog_text(source)
    else:
        if not os.path.exists(source):
            print(f"Error: File '{source}' not found.")
            exit(1)
            
        print(f"Reading from file: {source}")
        sample_blog = read_blog_from_file(source)
    
    if not sample_blog:
        print("Error: Could not read blog content from source.")
        exit(1)
    
    print(f"Analyzing content with {len(sample_blog)} characters and approximately {len(sample_blog.split())} words")

    analysis = analyze_blog(sample_blog)
    
    print("\n=== Blog Analysis Results ===")
    for key, value in analysis.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for subkey, subval in value.items():
                print(f"  {subkey}: {subval}")
        else:
            print(f"{key}: {value}")

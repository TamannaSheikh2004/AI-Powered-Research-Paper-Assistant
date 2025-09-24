#!/usr/bin/env python3
"""
Simple CLI version of the Research Assistant
Using OpenRouter (Grok) for text and Pollinations AI for images
"""

import os
import sys
import json
import requests
import urllib.parse
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

def check_api_setup():
    """Check API configuration"""
    print("🔑 Checking API setup...")
    
    if not OPENROUTER_API_KEY:
        print("❌ OPENROUTER_API_KEY missing in .env file")
        print("\n📝 Setup instructions:")
        print("1. Create a .env file in your project directory")
        print("2. Add: OPENROUTER_API_KEY=your_actual_key_here")
        print("3. Get your key at: https://openrouter.ai/keys")
        print("4. Pollinations AI is free - no key needed!")
        return False
    
    print("✅ OpenRouter API Key found")
    print("✅ Pollinations AI ready (no key required)")
    return True

def test_openrouter_connection():
    """Test OpenRouter API connection"""
    print("\n🧪 Testing OpenRouter (Grok) connection...")
    
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Research Assistant Test"
        }
        
        body = {
            "model": "x-ai/grok-beta",
            "messages": [{"role": "user", "content": "Hello! Just respond with 'API working'"}],
            "max_tokens": 20
        }
        
        response = requests.post(url, headers=headers, json=body, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        result = data["choices"][0]["message"]["content"].strip()
        print(f"✅ OpenRouter (Grok): {result}")
        return True
        
    except Exception as e:
        print(f"❌ OpenRouter test failed: {str(e)}")
        return False

def test_pollinations_connection():
    """Test Pollinations AI connection"""
    print("🧪 Testing Pollinations AI connection...")
    
    try:
        # Simple test image
        test_prompt = urllib.parse.quote("test image")
        url = f"https://image.pollinations.ai/prompt/{test_prompt}?width=100&height=100"
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        if response.headers.get('content-type', '').startswith('image/'):
            print("✅ Pollinations AI: Connection successful")
            return True
        else:
            print("❌ Pollinations AI: Unexpected response")
            return False
            
    except Exception as e:
        print(f"❌ Pollinations AI test failed: {str(e)}")
        return False

def generate_text_grok(prompt: str, max_tokens: int = 1500) -> str:
    """Generate text using OpenRouter's Grok model"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "AI Research Assistant CLI"
    }
    
    body = {
        "model": "x-ai/grok-beta",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert academic researcher. Generate high-quality, detailed, and specific research content."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(url, headers=headers, json=body, timeout=120)
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
        
    except Exception as e:
        return f"[Error generating content: {str(e)}]"

def generate_research_paper(topic: str):
    """Generate a comprehensive research paper"""
    print(f"\n🔬 Generating research paper on: '{topic}'")
    print("=" * 70)
    
    sections = {
        "Abstract": f"""Write a comprehensive abstract for a research paper on "{topic}".
        Include: background, objectives, methodology, key results, and conclusions.
        Length: 200-250 words. Be specific to the {topic} domain.""",
        
        "Introduction": f"""Write a detailed introduction for "{topic}" research.
        Cover: background, current state, challenges, research gap, objectives, contributions.
        Length: 400-500 words. Make it domain-specific and technical.""",
        
        "Literature Review": f"""Write a literature review for "{topic}".
        Discuss: existing approaches, methodologies, comparative analysis, identified gaps.
        Include hypothetical but realistic citations.
        Length: 300-400 words.""",
        
        "Methodology": f"""Write a methodology section for "{topic}" research.
        Describe: approach, system design, implementation, data collection, evaluation metrics.
        Length: 300-350 words. Be technical and specific.""",
        
        "Results": f"""Write a results section for "{topic}".
        Present: experimental outcomes, performance metrics, analysis, key findings.
        Length: 250-300 words.""",
        
        "Discussion": f"""Write a discussion for "{topic}" research.
        Analyze: result interpretation, implications, limitations, future work.
        Length: 200-250 words.""",
        
        "Conclusion": f"""Write a conclusion for "{topic}" research.
        Summarize: contributions, findings, impact, future directions.
        Length: 150-200 words."""
    }
    
    paper_content = []
    paper_content.append(f"# {topic}")
    paper_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    paper_content.append("=" * 70)
    
    for section_name, prompt in sections.items():
        print(f"📝 Generating: {section_name}...")
        
        content = generate_text_grok(prompt, max_tokens=1800)
        
        # Validate content
        if content.startswith("[Error"):
            print(f"   ❌ Failed: {content[:100]}...")
        else:
            preview = content[:120] + "..." if len(content) > 120 else content
            print(f"   ✅ Generated ({len(content)} chars): {preview}")
        
        paper_content.append(f"\n## {section_name.upper()}\n")
        paper_content.append(content)
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).strip()[:25]
    filename = f"research_paper_{safe_topic.replace(' ', '_')}_{timestamp}.txt"
    
    full_content = "\n".join(paper_content)
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(full_content)
        
        print(f"\n💾 Paper saved as: {filename}")
        print(f"📊 Total content: {len(full_content)} characters")
        print(f"📄 Sections generated: {len(sections)}")
        
        return filename
        
    except Exception as e:
        print(f"❌ Error saving file: {str(e)}")
        return None

def interactive_mode():
    """Interactive CLI mode"""
    print("\n🔬 AI Research Assistant - Interactive Mode")
    print("Using OpenRouter (Grok) + Pollinations AI")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Generate research paper")
        print("2. Test API connections")
        print("3. Quick topic suggestions")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            topic = input("\n📋 Enter your research topic: ").strip()
            if topic:
                filename = generate_research_paper(topic)
                if filename:
                    print(f"\n🎉 Success! Generated: {filename}")
                    
                    view = input("\n👀 View the content? (y/n): ").strip().lower()
                    if view in ['y', 'yes']:
                        try:
                            with open(filename, "r", encoding="utf-8") as f:
                                content = f.read()
                            print("\n" + "="*60)
                            print(content)
                            print("="*60)
                        except Exception as e:
                            print(f"Error reading file: {e}")
                else:
                    print("❌ Failed to generate paper")
            else:
                print("❌ Please enter a valid topic")
                
        elif choice == "2":
            print("\n🔧 Testing API connections...")
            openrouter_ok = test_openrouter_connection()
            pollinations_ok = test_pollinations_connection()
            
            if openrouter_ok and pollinations_ok:
                print("\n✅ All systems operational!")
            else:
                print("\n❌ Some services have issues")
                
        elif choice == "3":
            print("\n💡 Research topic suggestions:")
            suggestions = [
                "Artificial Intelligence in Healthcare Diagnostics",
                "Blockchain Technology for Supply Chain Transparency",
                "Machine Learning in Climate Change Prediction",
                "Quantum Computing Applications in Cryptography",
                "IoT Security in Smart Cities",
                "Natural Language Processing for Mental Health",
                "Computer Vision in Autonomous Vehicles",
                "Deep Learning for Drug Discovery"
            ]
            
            for i, suggestion in enumerate(suggestions, 1):
                print(f"{i}. {suggestion}")
            
            choice_num = input("\nSelect a topic (1-8) or press Enter to continue: ").strip()
            if choice_num.isdigit() and 1 <= int(choice_num) <= 8:
                selected_topic = suggestions[int(choice_num) - 1]
                print(f"\nSelected: {selected_topic}")
                
                confirm = input("Generate paper for this topic? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    filename = generate_research_paper(selected_topic)
                    if filename:
                        print(f"🎉 Generated: {filename}")
                        
        elif choice == "4":
            print("\n👋 Thanks for using AI Research Assistant!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

def main():
    """Main function"""
    print("🚀 AI Research Assistant CLI")
    print("Powered by OpenRouter (Grok) & Pollinations AI")
    print("-" * 50)
    
    # Check setup
    if not check_api_setup():
        sys.exit(1)
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        # Direct mode
        topic = " ".join(sys.argv[1:])
        print(f"\n🎯 Direct mode: '{topic}'")
        
        # Quick connection test
        if not test_openrouter_connection():
            print("❌ OpenRouter connection failed")
            sys.exit(1)
        
        filename = generate_research_paper(topic)
        if filename:
            print(f"\n🎉 Successfully generated: {filename}")
        else:
            print("❌ Generation failed")
            sys.exit(1)
    else:
        # Interactive mode
        interactive_mode()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)
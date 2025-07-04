import streamlit as st
import json
import time
import os
import uuid
import requests
import boto3
import nltk
from dotenv import load_dotenv
from openai import AzureOpenAI
from pathlib import Path

# Load environment variables
load_dotenv()

# Setup NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Azure OpenAI Client
client = AzureOpenAI(
    azure_endpoint= st.secrets["azure_api"]["AZURE_OPENAI_ENDPOINT"],
    api_key= st.secrets["azure_api"]["AZURE_OPENAI_API_KEY"],
    api_version="2024-02-01"
)

AZURE_TTS_URL = st.secrets["azure"]["AZURE_TTS_URL"]
AZURE_API_KEY = st.secrets["azure"]["AZURE_API_KEY"]

AWS_ACCESS_KEY = st.secrets["aws"]["AWS_ACCESS_KEY"]
AWS_SECRET_KEY = st.secrets["aws"]["AWS_SECRET_KEY"]
AWS_REGION = st.secrets["aws"]["AWS_REGION"]
AWS_BUCKET = st.secrets["aws"]["AWS_BUCKET"]
S3_PREFIX = st.secrets["aws"]["S3_PREFIX"]
CDN_BASE = st.secrets["aws"]["CDN_BASE"]

voice_options = {
    "1": "alloy",
    "2": "echo",
    "3": "fable",
    "4": "onyx",
    "5": "nova",
    "6": "shimmer"
}

# === Utility Functions ===
def extract_article(url):
    import newspaper
    article = newspaper.Article(url)
    article.download()
    article.parse()
    return article.title, article.summary, article.text

def get_sentiment(text):
    from textblob import TextBlob
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.2:
        return "positive"
    elif polarity < -0.2:
        return "negative"
    else:
        return "neutral"

def detect_category_and_subcategory(text):
    prompt = f"""
You are an expert news analyst.

Analyze the following news article and return:

1. category
2. subcategory
3. emotion

Article:
\"\"\"{text[:3000]}\"\"\"

Return as JSON:
{{
  "category": "...",
  "subcategory": "...",
  "emotion": "..."
}}
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Classify article into category, subcategory, and emotion."},
            {"role": "user", "content": prompt.strip()}
        ]
    )

    content = response.choices[0].message.content.strip()
    content = content.strip("```json").strip("```").strip()

    try:
        return json.loads(content)
    except:
        return {
            "category": "Unknown",
            "subcategory": "General",
            "emotion": "Neutral"
        }

def title_script_generator(category, subcategory, emotion, article_text, character_sketch=None):
    if not character_sketch:
        character_sketch = "Rohan Sharma is a sincere and articulate Hindi-English news anchor..."

    system_prompt = """
You are a digital content editor.

Create a structured 5-slide web story from the article below. Each slide must contain:
- A short English title (for the slide)
- A prompt: a clear instruction telling another GPT model what narration to write (don't write the narration here)

Format:
{
  "slides": [
    { "title": "...", "prompt": "..." },
    ...
  ]
}
"""
    user_prompt = f"""
Category: {category}
Subcategory: {subcategory}
Emotion: {emotion}

Article:
\"\"\"{article_text[:3000]}\"\"\"
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ]
    )

    content = response.choices[0].message.content.strip()
    content = content.strip("```json").strip("```").strip()

    try:
        slides_raw = json.loads(content)["slides"]
    except:
        return {"category": category, "subcategory": subcategory, "emotion": emotion, "slides": []}

    headline = article_text.split("\n")[0].strip().replace('"', '')
    slide1_script = f"Namaskar doston, main hoon Rohan Sharma. Aaj ki badi khabar: {headline}"

    slides = [{
        "title": headline[:80],
        "prompt": "Intro slide with greeting and headline.",
        "image_prompt": f"Vector-style illustration of Rohan Sharma presenting news: {headline}",
        "script": slide1_script
    }]

    for slide in slides_raw:
        narration_prompt = f"""
Write a 3–4 line Hindi-English narration in the voice of Rohan Sharma.

Instruction: {slide['prompt']}
Tone: Warm, simple, and clear. Avoid self-introduction.

Character sketch:
{character_sketch}
"""
        narration_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You write news narration in Hindi-English mix."},
                {"role": "user", "content": narration_prompt.strip()}
            ]
        )
        narration = narration_response.choices[0].message.content.strip()
        slides.append({
            "title": slide['title'],
            "prompt": slide['prompt'],
            "image_prompt": f"Modern vector-style visual for: {slide['title']}",
            "script": narration
        })

    return {
        "category": category,
        "subcategory": subcategory,
        "emotion": emotion,
        "slides": slides
    }

def restructure_slide_output(final_output):
    slides = final_output.get("slides", [])
    structured = {}
    for idx, slide in enumerate(slides):
        key = f"s{idx + 1}paragraph1"
        structured[key] = slide.get("script", "").strip()
    return structured

def synthesize_and_upload(paragraphs, voice):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )

    result = {}
    os.makedirs("temp", exist_ok=True)

    index = 2
    for text in paragraphs.values():
        st.write(f"🛠️ Processing: slide{index}")

        response = requests.post(
            AZURE_TTS_URL,
            headers={
                "Content-Type": "application/json",
                "api-key": AZURE_API_KEY
            },
            json={
                "model": "tts-1-hd",
                "input": text,
                "voice": voice
            }
        )
        response.raise_for_status()

        filename = f"tts_{uuid.uuid4().hex}.mp3"
        local_path = os.path.join("temp", filename)

        with open(local_path, "wb") as f:
            f.write(response.content)

        s3_key = f"{S3_PREFIX}{filename}"
        s3.upload_file(local_path, AWS_BUCKET, s3_key)
        cdn_url = f"{CDN_BASE}{s3_key}"

        slide_key = f"slide{index}"
        paragraph_key = f"s{index}paragraph1"
        audio_key = f"audio_url{index}"

        result[slide_key] = {
            paragraph_key: text,
            audio_key: cdn_url,
            "voice": voice
        }

        index += 1
        os.remove(local_path)

    return result

# === Streamlit UI ===
tab1, tab2 = st.tabs(["📰 Web Story Prompt Generator", "🔊 TTS + S3 Upload"])

with tab1:
    st.title("🧠 Generalized Web Story Prompt Generator")
    url = st.text_input("Enter a news article URL")
    persona = st.selectbox(
        "Choose audience persona:",
        ["genz", "millenial", "working professionals", "creative thinkers", "spiritual explorers"]
    )

    if url and persona:
        with st.spinner("Analyzing the article and generating prompts..."):
            try:
                title, summary, full_text = extract_article(url)
                sentiment = get_sentiment(summary)
                result = detect_category_and_subcategory(full_text)
                category, subcategory, emotion = result["category"], result["subcategory"], result["emotion"]

                output = title_script_generator(category, subcategory, emotion, full_text)
                final_output = {
                    "title": title,
                    "summary": summary,
                    "sentiment": sentiment,
                    "emotion": emotion,
                    "category": category,
                    "subcategory": subcategory,
                    "persona": persona,
                    "slides": output.get("slides", [])
                }

                st.success("✅ Prompt generation complete!")
                st.json(final_output)

                structured_output = restructure_slide_output(final_output)
                st.subheader("📄 Structured Slide Narration Format")
                st.json(structured_output)

                timestamp = int(time.time())
                filename = f"structured_slides_{timestamp}.json"
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(structured_output, f, indent=2, ensure_ascii=False)

                with open(filename, "r", encoding="utf-8") as f:
                    st.download_button(
                        label=f"⬇️ Download JSON ({timestamp})",
                        data=f.read(),
                        file_name=filename,
                        mime="application/json"
                    )
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

with tab2:
    st.title("🎙️ GPT-4o Text-to-Speech to S3")
    uploaded_file = st.file_uploader("Upload structured slide JSON", type=["json"])
    voice_label = st.selectbox("Choose Voice", list(voice_options.values()))

    if uploaded_file and voice_label:
        paragraphs = json.load(uploaded_file)
        st.success(f"✅ Loaded {len(paragraphs)} paragraphs")

        if st.button("🚀 Generate TTS + Upload to S3"):
            with st.spinner("Please wait..."):
                output = synthesize_and_upload(paragraphs, voice_label)
                st.success("✅ Done uploading to S3!")
                timestamp = int(time.time())
                output_filename = f"tts_output_{timestamp}.json"

                with open(output_filename, "w", encoding="utf-8") as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)

                with open(output_filename, "r", encoding="utf-8") as f:
                    st.download_button(
                        label="⬇️ Download Output JSON",
                        data=f.read(),
                        file_name=output_filename,
                        mime="application/json"
                    )

with tab3:
    st.title("🎞️ AMP Web Story Generator with Full Animation and Audio")

    # Fixed path to template
    TEMPLATE_PATH = Path("test.html")

    # Updated function with full AMP structure
    def generate_slide(paragraph: str, audio_url: str):
        return f"""
        <amp-story-page id="c29cbf94-847a-4bb7-a4eb-47d17d8c2d5a" auto-advance-after="page-c29cbf94-847a-4bb7-a4eb-47d17d8c2d5a-background-audio" class="i-amphtml-layout-container" i-amphtml-layout="container">
            <amp-story-animation layout="nodisplay" trigger="visibility" hidden="hidden">
                <script type="application/json">[{{"selector":"#anim-1a95e072-cada-435a-afea-082ddd65ff10","keyframes":{{"opacity":[0,1]}},"delay":0,"duration":600}}]</script>
            </amp-story-animation>
            <amp-story-animation layout="nodisplay" trigger="visibility" hidden="hidden">
                <script type="application/json">[{{"selector":"#anim-a938fe3f-03cf-47c5-9a84-da919c4f870b","keyframes":{{"transform":["translate3d(-115.2381%, 0px, 0)","translate3d(0px, 0px, 0)"]}},"delay":0,"duration":600}}]</script>
            </amp-story-animation>
            <amp-story-animation layout="nodisplay" trigger="visibility" hidden="hidden">
                <script type="application/json">[{{"selector":"#anim-f7c5981e-ac77-48d5-9b40-7a987a3e2ab0","keyframes":{{"opacity":[0,1]}},"delay":0,"duration":600}}]</script>
            </amp-story-animation>
            <amp-story-animation layout="nodisplay" trigger="visibility" hidden="hidden">
                <script type="application/json">[{{"selector":"#anim-0c1e94dd-ab91-415c-9372-0aa2e7e61630","keyframes":{{"transform":["translate3d(-115.55555%, 0px, 0)","translate3d(0px, 0px, 0)"]}},"delay":0,"duration":600}}]</script>
            </amp-story-animation>
            <amp-story-grid-layer template="vertical" aspect-ratio="412:618" class="grid-layer" style="--aspect-ratio:412/618;">
                <div class="_f09cc7b page-fullbleed-area">
                    <div class="page-safe-area">
                        <div class="_6120891">
                            <div class="_89d52dd mask" id="el-f00095ab-c147-4f19-9857-72ac678f953f">
                                <div class="_dc67a5c fill"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </amp-story-grid-layer>
            <amp-story-grid-layer template="fill">
                <amp-video autoplay layout="fixed" width="1" height="1" poster="" id="page-c29cbf94-847a-4bb7-a4eb-47d17d8c2d5a-background-audio" cache="google" style="width:1px;height:1px">
                    <source type="audio/mpeg" src="{audio_url}">
                </amp-video>
            </amp-story-grid-layer>
            <amp-story-grid-layer template="vertical" aspect-ratio="412:618" class="grid-layer" style="--aspect-ratio:412/618;">
                <div class="page-fullbleed-area">
                    <div class="page-safe-area">
                        <div class="_c19e533">
                            <div class="_89d52dd mask" id="el-344ed989-789b-4a01-a124-9ae1d15d67f4">
                                <div data-leaf-element="true" class="_8aed44c">
                                    <amp-img layout="fill" src="https://media.suvichaar.org/upload/polaris/polarisslide.png" alt="polarisslide.png" disable-inline-width="true"></amp-img>
                                </div>
                            </div>
                        </div>
                        <div class="_3d0c7a9">
                            <div id="anim-1a95e072-cada-435a-afea-082ddd65ff10" class="_75da10d animation-wrapper">
                                <div id="anim-a938fe3f-03cf-47c5-9a84-da919c4f870b" class="_e559378 animation-wrapper">
                                    <div id="el-2f080472-6c81-40a1-ac00-339cc8981388" class="_5342a26">
                                        <h3 class="_d1a8d0d fill text-wrapper"><span><span class="_14af73e">{paragraph}</span></span></h3>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="_a336742">
                            <div id="anim-f7c5981e-ac77-48d5-9b40-7a987a3e2ab0" class="_75da10d animation-wrapper">
                                <div id="anim-0c1e94dd-ab91-415c-9372-0aa2e7e61630" class="_09239f8 animation-wrapper">
                                    <div id="el-1a0d583c-c99b-4156-825b-3188408c0551" class="_ee8f788">
                                        <h2 class="_59f9bb8 fill text-wrapper"><span><span class="_14af73e"></span></span></h2>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </amp-story-grid-layer>
        </amp-story-page>
        """

    # Upload output.json
    output_json_file = st.file_uploader("📦 Upload Output JSON", type=["json"], key="amp_json_upload")

    if not TEMPLATE_PATH.exists():
        st.error("❌ 'test.html' not found in the repo directory.")
    else:
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            template_html = f.read()

        if output_json_file:
            try:
                output_data = json.load(output_json_file)

                if "<!--INSERT_SLIDES_HERE-->" not in template_html:
                    st.error("❌ Placeholder <!--INSERT_SLIDES_HERE--> not found in test.html.")
                else:
                    all_slides = ""
                    for key in sorted(output_data.keys(), key=lambda x: int(x.replace("slide", ""))):
                        slide_num = key.replace("slide", "")
                        data = output_data[key]
                        para_key = f"s{slide_num}paragraph1"
                        audio_key = f"audio_url{slide_num}"

                        if para_key in data and audio_key in data:
                            paragraph = data[para_key].replace("’", "'").replace('"', '&quot;')
                            audio_url = data[audio_key]
                            all_slides += generate_slide(paragraph, audio_url)

                    final_html = template_html.replace("<!--INSERT_SLIDES_HERE-->", all_slides)

                    timestamp = int(time.time())
                    filename = f"final_output_modified_{timestamp}.html"

                    st.success("✅ AMP HTML generated!")
                    st.download_button(
                        label="📥 Download Final AMP HTML",
                        data=final_html,
                        file_name=filename,
                        mime="text/html"
                    )

            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")

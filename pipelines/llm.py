import os
from openai import OpenAI
import json
import time
import re
from PIL import Image
import random
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from io import BytesIO
import base64
from datetime import datetime

openai_client = OpenAI()
GOOGLE_API_KEY=os.environ['GOOGLE_API_KEY']
gemini_client = genai.GenerativeModel('models/gemini-1.0-pro-vision-latest')
last_ask_time = time.time()

genai.configure(api_key=GOOGLE_API_KEY)

with open('assets/gemini_demon.txt', 'r') as f:
    gemini_templates = f.read()

with open('assets/gpt_demon.txt', 'r') as f:
    gpt_templates = f.read()

def format_check(array):
    # Check if the input is a list
    # Check if the list contains exactly one item
    # Check if the item in the list is either 0 or 1
    return isinstance(array, list) and len(array) == 1 and array[0] in [0, 1]


def parse_response(text):
    """
    Parse any array-like object from the text with re while the text may contains plain text
    """
    array_pattern = re.compile(r"\[.*?\]", re.DOTALL)
    array = array_pattern.findall(text)
    if array:
        return json.loads(array[0])
    else:
        raise Exception("Failed to parse the response.")

def base64encode_image(pil):
    # Input: PIL image
    # Output: base64 encoded string
    buffered = BytesIO()
    pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def ask_gpt(scenario, prompt, pils, model="gpt-4-turbo-2024-04-09"):
    if len(pils) != 2:
        raise ValueError("The number of pils should be 2.")
    prompt = gpt_templates.format(scenario=scenario, prompt=prompt)
    attempt = 0
    MAX_RETRIES = 3 # Due to budget constraints, we limit the number of retries to 3
    while attempt < MAX_RETRIES:
        try:
            time.sleep(1)
            response = openai_client.chat.completions.create(
                model=model,
                response_format={ "type": "json_object" },
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant to select the best image for a given scenario."},
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64encode_image(pils[0])}",
                                }
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64encode_image(pils[1])}",
                                }
                            },
                        ] 
                    }
                ],
                temperature=1.0
            ).choices[0].message.content
            print(response)
            
            
            data = json.loads(response)
            chosen_image = data["chosen_image"]
            with open(f'API_hist/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json', 'w') as f:
                json.dump({
                    'input': prompt,
                    'output': data,
                    'model': model,
                }, f, indent=4)
      
            if format_check(chosen_image):
                return [1 if i in chosen_image else 0 for i in range(2)]
            else:
                attempt += 1
        except Exception as e:
            breakpoint()
            print(e)
            attempt += 1
    

    print("Failed to get a valid response from the GPT model.")
    ret = [0, 1]
    random.shuffle(ret)
    return ret

def ask_gemini(scenario, prompt, pils):
    prompt = gemini_templates.format(scenario=scenario, prompt=prompt)
    attempt = 0
    MAX_RETRIES = 30
    global last_ask_time
    while attempt < MAX_RETRIES:
        try:
            # the request has 2 RPM limit
            # Wait until 31 second after the last request
            sleeptime = 31 - (time.time() - last_ask_time)
            if sleeptime > 0:
                time.sleep(sleeptime)
            response = gemini_client.generate_content(
                [prompt, *pils],
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=8192,
                    temperature=1.0
                    )
            ).text
            last_ask_time = time.time()
            print(response)
            
            array_like = parse_response(response)
            if format_check(array_like):
                return [1 if i in array_like else 0 for i in range(2)]
            else:
                attempt += 1
        except Exception as e:
            print(e)
            attempt += 1

    # Failed to get a valid response from the Gemini model, return random choice
    ret = [0, 1]
    random.shuffle(ret)
    return ret

if __name__ == "__main__":
    scenario = "You are a journalist who wants to add a visual teaser for your article to grab attention on social media or your news website."
    prompt = "A mysterious, glowing object discovered in an unexpected place, sparking curiosity and wonder. The setting changes based on the viewer's background, transforming the object's significance and the surrounding environment to match the realms of education, history, literature, design, science, and imagination."
    test_dir = '/scratch1/users/rareone0602/demon/experiments/teaser/2024-03-31_05-20-46/trajectory'
    pils = [Image.open(os.path.join(test_dir, files)) for files in os.listdir(test_dir)[:16]]
    ret = ask_gemini(scenario, prompt, pils)
    breakpoint()
    print(ret)
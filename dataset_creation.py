import argparse
import os 

import anthropic
import google.generativeai as genai
from groq import Groq
from openai import OpenAI
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt-4o', choices=["gpt4o", "llama70b", "llama8b", "sonnet", "qwen72b", "qwen32b", "gemini"])
parser.add_argument('--data_file', type=str)

MAX_OUTPUT_TOKENS = 20
TEMPERATURE = 0.7
args = parser.parse_args()

def gemini_greation(client, data):
    responses= []
    for input_text in tqdm(data):
        prompt = f"Complete the given sentence\n {input_text}"
        response = gemini.generate_content(prompt)   
        output = response.text
        responses.append(output)
    return responses

def anthropic_generation(client, model_name, data):
    responses= []
    for input_text in tqdm(data):
        messages = [
            {"role": "user", "content": [{"text": input_text, "type":"text"} ]}
        ]
        completion  = client.messages.create(
            model=model_name,
            system="Complete the given sentence.",
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS
        )
        output = completion.content[0].text.strip()
        responses.append(output)
    return responses


def api_based_generation(client, model_name, data):
    responses= []
    for input_text in tqdm(data):
        messages = [
            {"role": "system", "content": "Complete the given sentence."},
            {"role": "user", "content": input_text}
        ]
        completion  = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS
        )
        output = completion.choices[0].message.content.strip()
        responses.append(output)
    return responses

def local_generation(model_name, data):
    responses= []
    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for input_text in tqdm(data):
        messages = [
            {"role": "system", "content": "Complete the given sentence."},
            {"role": "user", "content": input_text}
        ]
        
        formatted_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
        )
        tokenized_input = tokenizer([formatted_input], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=MAX_OUTPUT_TOKENS
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        responses.append(response)
    return responses


openai_client = OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY')
)

groq_client = Groq(
    api_key=os.environ.get('GROQ_API_KEY')
)

anthropic_client = anthropic.Anthropic(
    api_key=os.environ.get('ANTHROPIC_API_KEY')
)

genai.configure(
    api_key=os.environ.get('GOOGLE_API_KEY')
)
SAFE = [
        {
            "category": "HARM_CATEGORY_DANGEROUS",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
]

gemini = genai.GenerativeModel('gemini-1.5-pro',safety_settings=SAFE, generation_config=genai.GenerationConfig(temperature=TEMPERATURE, max_output_tokens=MAX_OUTPUT_TOKENS)) 


model_mappings = {
    "gpt4o": "gpt-4o-2024-08-06",
    "llama70b": "llama-3.1-70b-versatile",
    "llama8b": "llama-3.1-8b-instant",
    "sonnet": "claude-3-5-sonnet-20240620",
    "qwen72b": "Qwen/Qwen2.5-72B-Instruct",
    "qwen32b": "Qwen/Qwen2.5-32B-Instruct",
    "gemini": "gemini-1.5-pro"
}
MODEL_NAME = model_mappings[args.model]
data = pd.read_csv(args.data_file)

if "Qwen" in MODEL_NAME:
    responses = local_generation(MODEL_NAME, data)
elif "gemini" in MODEL_NAME:
    responses = gemini_greation(gemini, data)
elif "llama" in MODEL_NAME:
    responses = api_based_generation(groq_client, MODEL_NAME, data)
elif "claude" in MODEL_NAME:
    responses = anthropic_generation(anthropic_client, MODEL_NAME, data)
elif "gpt" in MODEL_NAME:
    responses = api_based_generation(openai_client, MODEL_NAME, data)
else:
    raise ValueError(f"Model not supported {args.model}")

print(responses)
    


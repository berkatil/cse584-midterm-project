import argparse
import json
import os 
import time

import anthropic
import google.generativeai as genai
from google.api_core import retry
from google.generativeai.types import RequestOptions
from groq import Groq
from openai import OpenAI
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt4o', choices=["gpt4o", "llama70b", "llama8b", "sonnet", "qwen72b", "qwen32b", "gemini"])
parser.add_argument('--data_file', type=str)
#  dDONE: gp4o, sonnet, qwen32b, llama70b and qwen72b are running.
MAX_OUTPUT_TOKENS = 30
TEMPERATURE = 0.7
args = parser.parse_args()

def gemini_greation(client, input_text):
    #responses= []
    #for input_text in tqdm(data):
    prompt = f"Complete the given sentence. Do not give any explanation\n {input_text}"
    response = gemini.generate_content(prompt, request_options=RequestOptions(retry=retry.Retry(initial=10, multiplier=2, maximum=60, timeout=300)))   
    output = response.text
    return output
    #responses.append(output)
    #return responses

def anthropic_generation(client, model_name, input_text):
    #responses= []
    #for input_text in tqdm(data):
    messages = [
        {"role": "user", "content": [{"text": input_text, "type":"text"} ]}
    ]
    completion  = client.messages.create(
        model=model_name,
        system="Complete the given sentence. Do not give any explanation",
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_OUTPUT_TOKENS+10
    )
    output = completion.content[0].text.strip()
    return output
    #    responses.append(output)
    #return responses


def api_based_generation(client, model_name, input_text):
    #responses= []
    #for input_text in tqdm(data):
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
    return output
    #responses.append(output)
    #return responses

def local_generation(model_name, input_text):
    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    messages = [
        {"role": "system", "content": "Complete the given sentence."},
        {"role": "user", "content": input_text}
    ]
    
    formatted_input = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
    )
    model_inputs = tokenizer([formatted_input], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=MAX_OUTPUT_TOKENS
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response



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
with open(args.data_file) as f:
    data = json.load(f)
texts = []
indices = []
for x_i in data:
    texts.append(x_i["text"])
    indices.append(x_i["index"])

if os.path.exists(f"{args.model}.csv"):
    model_responses = pd.read_csv(f"{args.model}.csv")
else:
    model_responses = pd.DataFrame(columns=["index", "text", "response"])

already_processed_indices = model_responses["index"].tolist()

if "Qwen" in MODEL_NAME:
    for input_text, index in tqdm(zip(texts, indices)):
        if index in already_processed_indices:
            continue
        response = local_generation(MODEL_NAME, input_text)
        model_responses = model_responses._append({"index": index, "text": input_text, args.model: response}, ignore_index=True)
        model_responses.to_csv(f"{args.model}.csv", index=False)
        model_responses = pd.read_csv(f"{args.model}.csv")
elif "gemini" in MODEL_NAME:
    for input_text, index in tqdm(zip(texts, indices)):
        if index in already_processed_indices:
            continue
        response = gemini_greation(gemini, input_text)
        model_responses = model_responses._append({"index": index, "text": input_text, args.model: response}, ignore_index=True)
        model_responses.to_csv(f"{args.model}.csv", index=False)
        model_responses = pd.read_csv(f"{args.model}.csv")
        
    
elif "llama" in MODEL_NAME:
    for input_text, index in tqdm(zip(texts, indices)):
        if index in already_processed_indices:
            continue
        try:
            response = api_based_generation(groq_client, MODEL_NAME, input_text)
        except:
            try:
                response = api_based_generation(groq_client, MODEL_NAME, input_text)
            except:
                response = api_based_generation(groq_client, MODEL_NAME, input_text)
        model_responses = model_responses._append({"index": index, "text": input_text, args.model: response}, ignore_index=True)
        model_responses.to_csv(f"{args.model}.csv", index=False)
        model_responses = pd.read_csv(f"{args.model}.csv")
    
elif "claude" in MODEL_NAME:
    for input_text, index in tqdm(zip(texts, indices)):
        if index in already_processed_indices:
            continue
        if input_text == "":
            continue
        response = anthropic_generation(anthropic_client, MODEL_NAME, input_text)
        model_responses = model_responses._append({"index": index, "text": input_text, args.model: response}, ignore_index=True)
        model_responses.to_csv(f"{args.model}.csv", index=False)
        model_responses = pd.read_csv(f"{args.model}.csv")

elif "gpt" in MODEL_NAME:
    for input_text, index in tqdm(zip(texts, indices)):
        if index in already_processed_indices:
            continue
        response = api_based_generation(openai_client, MODEL_NAME, input_text)
        model_responses = model_responses._append({"index": index, "text": input_text, args.model: response}, ignore_index=True)
        model_responses.to_csv(f"{args.model}.csv", index=False)
        model_responses = pd.read_csv(f"{args.model}.csv")
else:
    raise ValueError(f"Model not supported {args.model}")



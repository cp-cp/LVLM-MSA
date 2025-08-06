import base64
import requests
import os
import json
import argparse

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_model_response(image_path, text, api_key, model_name="gpt-4o"):
    
    os.environ["http_proxy"] = "http://localhost:7890"
    os.environ["https_proxy"] = "http://localhost:7890"

    # Encode the image
    base64_image = encode_image(image_path)

    # Prepare request headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Prepare the payload
    payload = {
    "model": "gpt-4o-mini",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": text,
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 1024
    }

    # Make the request to the model API
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    # Check for errors in response
    if response.status_code != 200:
        raise ValueError(f"Error: {response.status_code}, {response.text}")

    # Parse and return the model's response
    return response.json()['choices'][0]['message']['content']

def evaluate_from_json(json_file_path, output_file_path, image_folder, api_key):
    
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for entry in data:
            question_id = entry.get('question_id')
            text = entry.get('text')
            image_filename = entry.get('image')
            image_path = os.path.join(image_folder, image_filename)

            print(f"Evaluating question_id: {question_id}")

            try:
                response = get_model_response(image_path, text, api_key)
                result={
                    "question_id": question_id,
                    "text": text,
                    "image": image_filename,
                    "response": response
                }
            except Exception as e:
                print(f"Error during evaluation of question_id {question_id}: {str(e)}")
                result={
                    "question_id": question_id,
                    "text": text,
                    "image": image_filename,
                    "response": f"Error: {str(e)}"
                }
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
            outfile.flush()
                
def main():
    parser = argparse.ArgumentParser(description="Evaluate sarcasm detection using GPT-4o-mini model from a JSON file.")
    parser.add_argument("--json_file", required=True, help="Path to the JSON file containing the evaluation data.")
    parser.add_argument("--image_folder", required=True, help="Folder where the images are stored.")
    parser.add_argument("--api_key", required=True, help="API key for accessing the model.")
    parser.add_argument("--output_file", required=True, help="Path to the output JSON file.")
    args = parser.parse_args()

    evaluate_from_json(args.json_file, args.output_file,args.image_folder, args.api_key)

if __name__ == "__main__":
    main()
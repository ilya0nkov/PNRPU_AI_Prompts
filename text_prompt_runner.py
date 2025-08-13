import json
from openai import OpenAI
from user_data import data_base_url, data_prompt_file_path, data_model_name


def load_prompt_from_json(prompt_file_path):
    with open(prompt_file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_response(base_url, messages, model, top_p=0.8):
    client = OpenAI(
        base_url=base_url,
        api_key="token-abc123"
    )
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        top_p=top_p,
        stream=True,
        stream_options={"include_usage": True}
    )
    return completion


def send_message(prompt_data, base_url, model_name):
    messages = []

    for entry in prompt_data:
        role = entry["role"]
        text = entry["text"]

        # Создаем ТОЛЬКО текстовый контент
        content = [{"type": "text", "text": text}]

        messages.append({
            "role": role,
            "content": content
        })

    completion = get_response(base_url=base_url, messages=messages, model=model_name)
    for chunk in completion:
        jtemp = json.loads(chunk.model_dump_json())
        if len(jtemp["choices"]) > 0 and "content" in jtemp["choices"][0]["delta"]:
            print(jtemp["choices"][0]["delta"]["content"], end="")


if __name__ == "__main__":
    prompt_file_path = data_prompt_file_path
    base_url = data_base_url
    model_name = data_model_name

    prompt_data = load_prompt_from_json(prompt_file_path)

    # Убраны image_paths из параметров
    send_message(prompt_data, base_url, model_name)
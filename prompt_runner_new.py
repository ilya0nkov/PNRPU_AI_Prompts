import json
import base64
import os
import shutil
from pathlib import Path
from openai import OpenAI
from collections.abc import Iterable
from pdf2image import convert_from_path


def pdf_to_jpg(image_path):
    if image_path.endswith(".pdf"):
        pages = convert_from_path(image_path)
        pages[0].save(os.path.join(os.path.dirname(image_path), f'{Path(image_path).stem}.jpg'), 'JPEG')


def encode_image(image_path):
    # Конвертируем изображение в base64 для передачи в запросе
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded_image}"


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


def send_message(prompt_data, request_image_path, base_url, model_name, answer_file_path=None):
    messages = []

    if isinstance(prompt_data, dict):
        prompt_data = [prompt_data]

    # Обработка каждого элемента в `prompt_data`
    for entry in prompt_data:
        role = entry["role"] if "role" in entry else None
        text = entry["text"] if "text" in entry else None
        image_path = entry["image_path"] if "image_path" in entry else None
        top_p = float(entry["top_p"]) if "top_p" in entry else 0.8

        if not role:
            role = "user"
            print("WARNING: роль (role) не была найдена в json файле prompt-а. Используется 'user'.")
        if not text:
            print("ERROR: text не был найден. Запрос не отправлен.")
            return 1

        # Создаем сообщение с текстом
        content = [{"type": "text", "text": text}]

        # Если роль — `user`, добавляем изображения
        if role == "user" and image_path:
            base64_image = encode_image(image_path)
            content.append({"type": "image_url", "image_url": {"url": base64_image}})

        # Добавляем сообщение с ролью и контентом
        messages.append({
            "role": role,
            "content": content
        })

    # Добавляем запрашивоемое изображение в последнее сообщение
    if messages[-1]["role"] == "user":
        base64_image = encode_image(request_image_path)
        messages[-1]["content"].append({"type": "image_url", "image_url": {"url": base64_image}})

    # Отправляем запрос к модели
    answer = ""
    completion = get_response(base_url=base_url, messages=messages, model=model_name, top_p=top_p)
    for chunk in completion:
        jtemp = json.loads(chunk.model_dump_json())
        if len(jtemp["choices"]) > 0:
            answer += jtemp["choices"][0]["delta"]["content"]
            print(jtemp["choices"][0]["delta"]["content"], end="")
    print()
    if answer_file_path:
        os.makedirs(os.path.dirname(answer_file_path), exist_ok=True)
        with open(answer_file_path, 'w') as f:
            f.write(answer)


if __name__ == "__main__":
    # Пути и настройки
    base_url = "http://10.66.80.3:9000/v1"
    model_name = "Qwen/Qwen2-VL-72B-Instruct-AWQ"
    root_path = r"C:\Users\Mobil\Desktop\repo\PyCharmProjects\PNRPU_AI_prompts_auto\prompts\Onkov"
    test_images_path = os.path.join(root_path, "data", "test", "drawings")
    prompts_path = os.path.join(root_path, "prompts")
    prompts_owners = [x for x in os.listdir(prompts_path) if os.path.isdir(os.path.join(prompts_path, x))]
    test_images = [x for x in os.listdir(test_images_path) if x.endswith(tuple(['.jpg', '.png', '.jpeg']))]

    for owner in prompts_owners:
        print(f"{owner}:")
        for prompt_name in sorted(x for x in os.listdir(os.path.join(prompts_path, owner)) if x.endswith('.json')):
            print(f"\t{prompt_name}")
            for test_img in test_images:
                prompt_data = load_prompt_from_json(os.path.join(prompts_path, owner, prompt_name))
                send_message(
                    prompt_data,
                    os.path.join(test_images_path, test_img),
                    base_url,
                    model_name,
                    os.path.join(prompts_path, owner, "responses",
                                 Path(prompt_name).stem, f"{Path(test_img).stem}.txt")
                )

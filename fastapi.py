from fastapi import FastAPI, Request
import httpx

app = FastAPI()

# URL внешнего сервиса
EXTERNAL_SERVICE_URL = "https://example.com/external-api"

@app.post("/proxy")
async def proxy_request(request: Request):
    """
    Прокси-эндпоинт: принимает запрос от клиента,
    пересылает его во внешний сервис и возвращает ответ.
    """
    # Получение тела запроса
    body = await request.body()

    # Отправка запроса во внешний сервис
    async with httpx.AsyncClient() as client:
        external_response = await client.post(
            EXTERNAL_SERVICE_URL,
            content=body,  # Пересылаем тело запроса
            headers=dict(request.headers),  # Пересылаем заголовки
        )

    # Возврат ответа клиенту
    return {
        "body": external_response.json(),
    }

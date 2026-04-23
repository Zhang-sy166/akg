from langchain_anthropic import ChatAnthropic
import httpx


model_params = {
    'anthropic_api_url': 'https://gw.claudeapi.com',
    'model': "claude-opus-4-7",
    'max_tokens': 128,
    'temperature': 0.1,
    'top_p': 0.9,
    # 'frequency_penalty': 0.5,
    # 'presence_penalty': 0.5,
}
timeout = httpx.Timeout(60, read=60 * 10)
model = ChatAnthropic(
    api_key="",
    # http_client=httpx.Client(verify=False, timeout=timeout),
    # http_async_client=httpx.AsyncClient(verify=False, timeout=timeout),
    **model_params
)

# 调用模型
response = model.invoke("你好,你是谁，你的模型型号是哪个？")
print(response.content)

import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain.chat_models import init_chat_model

# 加载当前目录 .env 文件
load_dotenv()

# 获取 API Key
api_key = os.getenv("DEEPSEEK_API_KEY")

# 创建模型
llm = ChatDeepSeek(
    api_key=api_key,
    model="deepseek-v4-flash",
    temperature=0,      # 控制随机性，越低结果越稳定
    max_tokens=None,
    timeout=None,
    max_retries=2       # 失败后的最大重试次数
)

# 调用模型
resp = llm.invoke("你好，请介绍LangChain")
print(resp.content)

# # 指定了model，返回固定模型
# model = init_chat_model("deepseek-v4-flash", temperature=0.7)
# resp = model.invoke("你好，请介绍LangChain")
# print(resp.content)

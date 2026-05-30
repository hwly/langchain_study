import os
from langchain_openai import ChatOpenAI

# 建议将 API Key 放在环境变量中，或者直接在此处赋值（生产环境请勿硬编码）
# os.environ["DEEPSEEK_API_KEY"] = "sk-你的DeepSeek密钥"

# 初始化模型
llm = ChatOpenAI(
    model="deepseek-v4-pro",             # DeepSeek V4 模型名称
    openai_api_key="sk-708b1e18f698455fbdefd5369e221c80",        # 填入你的 DeepSeek API Key
    openai_api_base="https://api.deepseek.com", # DeepSeek 的 API 地址
    temperature=0.7,
    max_tokens=1024
)

# 测试一下
response = llm.invoke("你好，DeepSeek！请做一个简短的自我介绍。")
print(response.content)
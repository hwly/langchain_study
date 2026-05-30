import os
from dotenv import load_dotenv

from langchain_deepseek import ChatDeepSeek     # LangChain的DeepSeek聊天模型类
from langchain_core.messages import HumanMessage, SystemMessage     # SystemMessage，系统提示词，用于设定AI身份

# 加载当前目录 .env 文件
load_dotenv()

# 获取 API Key
api_key = os.getenv("DEEPSEEK_API_KEY")

# 创建DeepSeek模型
llm = ChatDeepSeek(
    api_key=api_key,
    model="deepseek-v4-flash",
    temperature=0.7,
    max_tokens=1024,
    timeout=60,
    max_retries=3
)

# 构造聊天消息
message = [
    SystemMessage(
        content="你是一名专业Python教师。"
    ),
    HumanMessage(
        content="请解释什么是LangChain，并给出简单示例。"
    )
]

# 调用模型
resp = llm.invoke(message)

# 输出结果
print("AI回复：", resp.content)

print('================================================')


# LangChain支持DeepSeek流式输出，需要使用deepseek-v4-pro推理模型
llm_pro = ChatDeepSeek(
    api_key=api_key,
    model="deepseek-v4-pro",
    temperature=0.7,
    max_tokens=1024,
    timeout=60,
)

for chunk in llm.stream("请介绍Python"):
    print(chunk.content, end='', flush=True)

print('================================================')

# 使用PromptTemplate（提示词模板）
## 结合PromptTemplate动态生成Prompt
from langchain_core.prompts import ChatPromptTemplate

prompt =  ChatPromptTemplate.from_template(
    "请详细解释：{topic}"
)

# | 管道符
chain = prompt | llm_pro

resp_prom = chain.invoke({
    "topic": "Transformer"
})

print(resp_prom.content)










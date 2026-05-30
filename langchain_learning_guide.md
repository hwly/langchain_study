# LangChain 学习清单

## 学习路线图

```
基础准备 → 核心概念 → Chains → RAG检索增强 → Agents智能体 → Memory记忆 → 进阶项目
```

---

## 阶段一：基础准备（1-2天）

### 1.1 环境搭建

```bash
pip install langchain langchain-openai langchain-community
pip install chromadb faiss-cpu   # 向量数据库
pip install python-dotenv
```

```python
# .env 文件
OPENAI_API_KEY=sk-xxxx
```

### 1.2 最简调用 — Hello LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
response = llm.invoke("用一句话介绍LangChain")
print(response.content)
```

> **目标**：跑通第一次调用，确认API可用。

---

## 阶段二：核心概念（2-3天）

### 2.1 Prompt Template — 提示词模板

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}，用专业但易懂的语言回答。"),
    ("human", "{question}"),
])

chain = prompt | llm

result = chain.invoke({
    "role": "Python高级工程师",
    "question": "装饰器是什么？"
})
print(result.content)
```

> **要点**：`{变量}` 占位符，`|` 管道符串联组件（LCEL语法）。

### 2.2 Output Parser — 输出解析

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# 方式1：纯文本解析
chain = prompt | llm | StrOutputParser()

# 方式2：JSON结构化输出
class Movie(BaseModel):
    name: str = Field(description="电影名称")
    year: int = Field(description="上映年份")
    rating: float = Field(description="评分")

parser = JsonOutputParser(pydantic_object=Movie)

prompt2 = ChatPromptTemplate.from_messages([
    ("system", "你是一个电影专家。\n{format_instructions}"),
    ("human", "{query}"),
])

chain2 = prompt2 | llm | parser

result = chain2.invoke({
    "query": "推荐一部科幻电影",
    "format_instructions": parser.get_format_instructions(),
})
print(result)
# {'name': 'Interstellar', 'year': 2014, 'rating': 9.4}
```

> **要点**：让LLM输出结构化数据，不用手动解析。

### 2.3 LCEL（LangChain Expression Language）

```python
# LCEL 核心思想：用 | 管道串联所有组件
chain = prompt | llm | StrOutputParser()

# 等价于手动调用
# formatted = prompt.invoke({...})
# response = llm.invoke(formatted)
# parsed = parser.invoke(response)

# 支持批量
results = chain.batch([
    {"role": "老师", "question": "什么是引力？"},
    {"role": "医生", "question": "什么是血压？"},
])

# 支持流式输出
for chunk in chain.stream({"role": "老师", "question": "讲个故事"}):
    print(chunk, end="", flush=True)
```

> **要点**：LCEL是LangChain的精髓，`prompt | llm | parser` 三件套。

---

## 阶段三：Chains 链式调用（2-3天）

### 3.1 Sequential Chain — 顺序链

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 第一步：生成菜名
prompt1 = ChatPromptTemplate.from_template(
    "根据食材{ingredients}，推荐一道菜名，只要菜名。"
)
# 第二步：根据菜名生成菜谱
prompt2 = ChatPromptTemplate.from_template(
    "写出{dish}的详细做法，包括步骤和用料。"
)

chain1 = prompt1 | llm | StrOutputParser()

# 用RunnablePassthrough传递上一步结果
from langchain_core.runnables import RunnablePassthrough

full_chain = (
    {"dish": chain1}
    | prompt2
    | llm
    | StrOutputParser()
)

result = full_chain.invoke({"ingredients": "鸡蛋、番茄"})
print(result)
```

### 3.2 带分支的链 — Router

```python
from langchain_core.runnables import RunnableLambda

def route_topic(input_dict):
    topic = input_dict["topic"]
    if topic == "数学":
        return "你是一位数学教授，用公式和逻辑解释问题。"
    elif topic == "历史":
        return "你是一位历史学家，结合时代背景解释事件。"
    else:
        return "你是一个通用助手。"

prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    ("human", "{question}"),
])

chain = (
    {
        "system_prompt": lambda x: route_topic(x),
        "question": lambda x: x["question"],
    }
    | prompt
    | llm
    | StrOutputParser()
)

print(chain.invoke({"topic": "数学", "question": "什么是微积分？"}))
```

> **要点**：链可以嵌套、分支、条件路由。

---

## 阶段四：RAG 检索增强生成（3-4天）⭐重点

### 4.1 文档加载与切分

```python
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载文档
loader = TextLoader("company_faq.txt", encoding="utf-8")
docs = loader.load()

# 切分
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,     # 每块最大字符数
    chunk_overlap=50,   # 块之间重叠字符数
)
chunks = splitter.split_documents(docs)
print(f"共 {len(chunks)} 个文本块")
```

### 4.2 向量化与存储

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 创建向量库
vectorstore = FAISS.from_documents(chunks, embeddings)

# 保存到本地
vectorstore.save_local("faiss_index")

# 加载
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
```

### 4.3 完整RAG流程

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_prompt = ChatPromptTemplate.from_template("""根据以下参考资料回答问题。如果资料中没有相关信息，请说"我不确定"。

参考资料：
{context}

问题：{question}
""")

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("公司的请假制度是什么？")
print(answer)
```

### 4.4 支持对话的RAG（多轮）

```python
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", "根据聊天历史和最新问题，生成一个独立的检索查询。"),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "根据以下资料回答问题：\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

chat_history = []

# 使用时需要维护chat_history列表
# 每轮对话后手动append
```

> **阶段目标**：完成一个能读取本地文档、回答问题的RAG系统。

---

## 阶段五：Agents 智能体（3-4天）⭐重点

### 5.1 什么是Agent

```
用户提问 → LLM判断需要什么工具 → 调用工具 → 拿到结果 → LLM总结回答
```

### 5.2 自定义Tool

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气（模拟）"""
    weather_data = {
        "北京": "晴天，15°C",
        "上海": "多云，18°C",
        "广州": "小雨，22°C",
    }
    return weather_data.get(city, f"{city}暂无天气数据")

@tool
def calculator(expression: str) -> str:
    """计算数学表达式，如 '2+3*4'"""
    try:
        result = eval(expression)  # 生产环境请用ast.literal_eval
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{e}"

@tool
def search_database(query: str) -> str:
    """在公司数据库中搜索信息"""
    # 这里替换为真实的数据库查询
    fake_db = {
        "张三": "张三，技术部，高级工程师",
        "李四": "李四，市场部，经理",
    }
    for key, val in fake_db.items():
        if key in query:
            return val
    return "未找到相关信息"
```

### 5.3 创建Agent

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

tools = [get_weather, calculator, search_database]

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能助手，可以使用工具来帮助用户。"),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,          # 打印中间过程
    max_iterations=5,      # 最大工具调用次数
)
```

### 5.4 运行Agent

```python
result = agent_executor.invoke({"input": "北京天气怎么样？"})
# > Entering new AgentExecutor chain...
# > 调用 get_weather("北京")
# > "北京：晴天，15°C"

result = agent_executor.invoke({"input": "(15 + 27) * 3 等于多少？"})
# > 调用 calculator("(15+27)*3")

result = agent_executor.invoke({"input": "帮我查一下张三的信息，然后算算 100 * 50"})
# > 调用两个工具

print(result["output"])
```

### 5.5 Structured Chat Agent（结构化工具）

```python
from langchain.agents import create_structured_chat_agent

# 适合需要复杂参数的工具
@tool
def search_product(
    name: str = "",
    min_price: float = 0,
    max_price: float = 999999,
    category: str = "",
) -> str:
    """搜索商品
    Args:
        name: 商品名称关键词
        min_price: 最低价格
        max_price: 最高价格
        category: 商品类别
    """
    return f"搜索结果：name={name}, 价格区间={min_price}-{max_price}, 类别={category}"
```

---

## 阶段六：Memory 记忆（1-2天）

### 6.1 对话记忆

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

# 简单的内存存储
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的助手。"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

chain = prompt | llm | StrOutputParser()

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# 多轮对话
config = {"configurable": {"session_id": "user1"}}

r1 = chain_with_history.invoke({"question": "我叫小明"}, config=config)
print(r1)

r2 = chain_with_history.invoke({"question": "我叫什么？"}, config=config)
print(r2)  # 会记住"小明"
```

### 6.2 总结式记忆（长对话）

```python
from langchain_core.prompts import ChatPromptTemplate

summary_prompt = ChatPromptTemplate.from_template("""
逐步总结以下对话内容，在已有总结基础上添加新信息：

已有总结：{summary}

新对话：
Human: {new_input}
AI: {new_response}

更新后的总结：
""")
```

---

## 阶段七：回调与调试（1天）

### 7.1 Callbacks

```python
from langchain_core.callbacks import BaseCallbackHandler

class MyCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"[LLM开始] prompts数量: {len(prompts)}")

    def on_llm_end(self, response, **kwargs):
        print(f"[LLM结束] token用量: {response.llm_output}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"[工具调用] {serialized['name']}: {input_str}")

    def on_chain_end(self, outputs, **kwargs):
        print(f"[链结束] 输出: {outputs}")

# 使用
result = chain.invoke({"question": "hello"}, config={"callbacks": [MyCallback()]})
```

---

## 阶段八：实战项目（按难度递进）

### 项目1：文档问答机器人 ⭐入门

```
技术栈：TextLoader + FAISS + RAG Chain
功能：上传PDF/TXT → 提问 → 基于文档回答
```

```python
# 核心代码（阶段四已有），扩展：
# 1. 支持多文档
# 2. 加上Gradio/Streamlit前端
# 3. 加上对话历史
```

### 项目2：多工具智能助手 ⭐⭐进阶

```
技术栈：Agent + 自定义Tools + Memory
功能：天气查询 + 计算器 + 数据库查询 + 网页搜索
```

```python
# 核心代码（阶段五已有），扩展：
# 1. 加入Tavily搜索工具
# 2. 加入SQL数据库工具
# 3. 持久化对话记忆
```

### 项目3：自动化工作流 ⭐⭐⭐高级

```
技术栈：Agent + RAG + 多轮Memory + LangGraph
功能：自动分析报告 → 检索历史数据 → 生成总结 → 发送邮件
```

```python
# LangGraph 示例（状态机式Agent）
# pip install langgraph

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_step: str

def researcher(state):
    # 研究节点：调用搜索工具
    return {"messages": [...], "next_step": "writer"}

def writer(state):
    # 写作节点：生成报告
    return {"messages": [...], "next_step": END}

graph = StateGraph(AgentState)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_edge("researcher", "writer")
graph.add_edge("writer", END)
graph.set_entry_point("researcher")

app = graph.compile()
result = app.invoke({"messages": ["帮我写一份市场分析报告"], "next_step": ""})
```

---

## 学习资源

| 资源 | 说明 |
|------|------|
| [LangChain官方文档](https://python.langchain.com/) | 最权威，优先看 |
| [LangChain GitHub](https://github.com/langchain-ai/langchain) | 看examples目录 |
| [LangSmith](https://smith.langchain.com/) | 在线调试/追踪平台 |
| [LangGraph文档](https://langchain-ai.github.io/langgraph/) | 多Agent/复杂工作流 |

---

## 每日学习计划

| 天数 | 内容 | 产出 |
|------|------|------|
| Day 1-2 | 环境搭建 + Prompt + Parser | 能跑通prompt\|llm\|parser |
| Day 3-4 | Chains + LCEL语法 | 能写多步骤链 |
| Day 5-7 | RAG全流程 | 做出文档问答demo |
| Day 8-10 | Agent + Tool | 做出多工具助手 |
| Day 11-12 | Memory + 回调 | 支持多轮对话 |
| Day 13-15 | 实战项目 | 完整项目1-2个 |

> **建议**：每学一个概念就写一个小脚本验证，不要只看不练。官方文档的 **Tutorials** 部分值得逐个跟做。

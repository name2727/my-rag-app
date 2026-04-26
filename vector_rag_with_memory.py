import streamlit as st
import requests
import jieba
import os

# 配置
api_key = st.secrets["DEEPSEEK_API_KEY"] if "DEEPSEEK_API_KEY" in st.secrets else ""

# 加载知识库文件
def load_knowledge_base():
    knowledge_dir = "knowledge_base"
    docs = {}
    if os.path.exists(knowledge_dir):
        for filename in os.listdir(knowledge_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(knowledge_dir, filename), "r", encoding="utf-8") as f:
                    docs[filename] = f.read().strip()
    return docs

# 关键词检索（返回最相关的内容片段）
def retrieve(question, docs):
    keywords = [w for w in jieba.lcut(question) if len(w) > 1]
    best_content = ""
    best_score = 0
    best_source = ""
    for filename, content in docs.items():
        score = sum(1 for kw in keywords if kw in content)
        if score > best_score:
            best_score = score
            best_content = content
            best_source = filename
    if best_score > 0:
        return best_content, best_source
    else:
        return None, None

# 调用 DeepSeek API
def ask_llm(question, context):
    if not api_key:
        return "请先在 Streamlit Secrets 中配置 DEEPSEEK_API_KEY"
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    prompt = f"请根据以下信息回答问题。如果信息不足，请说明。\n\n信息：{context}\n\n问题：{question}\n回答："
    data = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "stream": False}
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"API 错误：{response.status_code}"
    except Exception as e:
        return f"网络错误：{str(e)}"

# Streamlit UI
st.set_page_config(page_title="知识库问答", page_icon="📚")
st.title("📚 知识库问答助手")

# 加载知识库
docs = load_knowledge_base()
if not docs:
    st.warning("未找到知识库文件，请确保 `knowledge_base` 文件夹中存在 .txt 文件。")
else:
    st.success(f"已加载 {len(docs)} 个知识文件")

# 用户输入
question = st.text_input("请输入你的问题：")
if st.button("提问") and question:
    with st.spinner("检索中..."):
        context, source = retrieve(question, docs)
        if not context:
            st.info("未找到相关信息，请尝试其他问题。")
        else:
            st.write(f"**来源：{source}**")
            with st.spinner("生成回答中..."):
                answer = ask_llm(question, context[:2000])  # 限制长度
            st.write(f"**回答：** {answer}")

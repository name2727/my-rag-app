# app_kw.py - 关键词检索版，无向量数据库依赖
import streamlit as st
import requests
import jieba
import os

# 配置
KNOWLEDGE_DIR = "knowledge_base"
api_key = st.secrets["DEEPSEEK_API_KEY"]  # 从 Streamlit Secrets 读取

# 加载知识库
@st.cache_resource
def load_knowledge():
    chunks = []  # 每个元素为 (文本, 来源)
    if not os.path.exists(KNOWLEDGE_DIR):
        os.makedirs(KNOWLEDGE_DIR)
        return chunks
    for filename in os.listdir(KNOWLEDGE_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(KNOWLEDGE_DIR, filename), "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    # 简单分块（按段落）
                    paras = [p.strip() for p in content.split("\n\n") if p.strip()]
                    for para in paras:
                        chunks.append({"text": para, "source": filename})
    return chunks

def retrieve(question, chunks):
    keywords = [w for w in jieba.lcut(question) if len(w) > 1]
    best_chunk = None
    best_score = 0
    for chunk in chunks:
        score = sum(1 for kw in keywords if kw in chunk["text"])
        if score > best_score:
            best_score = score
            best_chunk = chunk
    return best_chunk

def ask_llm(question, context):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    prompt = f"请根据以下信息回答问题。如果信息不足，请明确说明。\n\n信息：{context}\n\n问题：{question}\n回答："
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"API错误：{response.status_code}"
    except Exception as e:
        return f"网络错误：{str(e)}"

# Streamlit 界面
st.set_page_config(page_title="知识库问答助手", page_icon="📚")
st.title("📚 知识库问答助手（关键词检索版）")

# 加载知识库
chunks = load_knowledge()
if not chunks:
    st.warning(f"请在 `{KNOWLEDGE_DIR}` 文件夹下放入一些 txt 文件。")
    st.stop()
st.success(f"已加载 {len(chunks)} 个知识片段")

# 聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("请输入问题："):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 检索
    best = retrieve(prompt, chunks)
    if not best:
        answer = "未找到相关信息，请尝试其他问题。"
        source = None
    else:
        source = best["source"]
        answer = ask_llm(prompt, best["text"])
        answer = f"**📄 来源：{source}**\n\n{answer}"
    
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

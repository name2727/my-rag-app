import os
import jieba
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions

# ========== 配置 ==========
KNOWLEDGE_DIR = "knowledge_base"
api_key = "sk-14f508359c934348b33f0d7be2fb52e1"  # 请替换
MODEL_PATH = r'D:\my_models\paraphrase-multilingual-MiniLM-L12-v2'  # 本地模型路径
# =========================

# 加载 embedding 模型
print("正在加载 embedding 模型...")
embedder = SentenceTransformer(MODEL_PATH)
print("模型加载完成")

# 加载知识库文件
def load_knowledge_base():
    documents = []
    if not os.path.exists(KNOWLEDGE_DIR):
        os.makedirs(KNOWLEDGE_DIR)
        return documents
    for filename in os.listdir(KNOWLEDGE_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(KNOWLEDGE_DIR, filename), "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    documents.append({"filename": filename, "content": content})
    return documents

# 分块
def chunk_document(content, filename, chunk_size=300):
    chunks = []
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append({"source": filename, "text": para})
        else:
            for i in range(0, len(para), chunk_size):
                chunk_text = para[i:i+chunk_size]
                chunks.append({"source": filename, "text": chunk_text})
    return chunks

# 初始化向量数据库
def build_vector_store(chunks):
    client = chromadb.Client()
    collection = client.create_collection(name="knowledge", embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_PATH))
    ids = []
    texts = []
    metadatas = []
    for i, chunk in enumerate(chunks):
        ids.append(str(i))
        texts.append(chunk["text"])
        metadatas.append({"source": chunk["source"]})
    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    return collection

# 检索
def retrieve(query, collection, top_k=1):
    results = collection.query(query_texts=[query], n_results=top_k)
    if results['documents'] and results['documents'][0]:
        return results['documents'][0][0], results['metadatas'][0][0]['source']
    else:
        return None, None

# 调用 API（带历史记录）
def ask_llm_with_history(question, context, history):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    # 构建当前用户消息（包含检索上下文）
    user_content = f"【参考信息】\n{context}\n\n【问题】\n{question}" if context else question
    # 将当前用户消息加入历史
    history.append({"role": "user", "content": user_content})
    
    # 如果历史太长，只保留最近 10 条（防止超出 token 限制）
    if len(history) > 10:
        # 保留 system 消息和最近 9 条
        history = [history[0]] + history[-9:]
    
    data = {
        "model": "deepseek-chat",
        "messages": history,
        "stream": False
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            answer = response.json()["choices"][0]["message"]["content"]
            # 将 AI 回答加入历史
            history.append({"role": "assistant", "content": answer})
            return answer, history
        else:
            return f"API错误：{response.status_code}", history
    except Exception as e:
        return f"网络错误：{str(e)}", history

# 主程序
def main():
    print("正在加载知识库...")
    docs = load_knowledge_base()
    if not docs:
        print(f"未在 {KNOWLEDGE_DIR} 中找到 txt 文件，请添加后重试。")
        return
    
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc["content"], doc["filename"])
        all_chunks.extend(chunks)
    print(f"共 {len(all_chunks)} 个文本块，正在生成向量...")
    collection = build_vector_store(all_chunks)
    print("向量数据库构建完成！")
    
    # 初始化对话历史（包含系统提示）
    conversation_history = [
        {"role": "system", "content": "你是一个基于知识库的问答助手。请根据【参考信息】回答问题。如果信息不足，请说明。不要编造答案。请结合对话历史理解用户问题。"}
    ]
    
    print("向量检索问答机器人已启动（支持多轮对话记忆，输入'退出'结束）\n")
    while True:
        question = input("你：")
        if question == "退出":
            print("再见！")
            break
        if not question.strip():
            continue
        
        # 检索
        context, source = retrieve(question, collection)
        if not context:
            print("AI：未找到相关信息。")
            continue
        
        print(f"[来源：{source}]")
        answer, conversation_history = ask_llm_with_history(question, context, conversation_history)
        print(f"AI：{answer}")

if __name__ == "__main__":
    main()
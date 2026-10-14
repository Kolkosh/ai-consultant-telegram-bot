import json
import os
from typing import List, Dict, Any
from celery import Celery
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
import chromadb
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest
from worker.tools import TOOLS_SCHEMA, AVAILABLE_TOOLS

# =====================================================================
# 1. Конфигурация и инициализация клиентов
# =====================================================================
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

app = Celery("tasks", broker=REDIS_URL, backend=REDIS_URL)

# OpenRouter SDK (совместим с OpenAI)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Подключение к хранилищу ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="store_knowledge_base")

# Инициализация FlashRank реранкера
try:
    ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
except Exception as e:
    print(f"Предупреждение: FlashRank не смог загрузиться ({e}). Будет использован фолбэк-реранкинг.")
    ranker = None


# =====================================================================
# 2. Модуль Advanced RAG (Hybrid Search + RRF + FlashRank)
# =====================================================================
def _get_all_chroma_docs():
    """Извлекает все документы из ChromaDB для построения корпуса BM25."""
    res = collection.get(include=["documents", "metadatas"])
    return res.get("documents", []), res.get("metadatas", []), res.get("ids", [])


def hybrid_search(query: str, top_k: int = 6) -> List[Dict[str, Any]]:
    """
    Гибридный поиск:
    1. Векторный поиск ChromaDB (Dense Retrieval)
    2. Поиск по ключевым словам BM25 (Sparse Retrieval)
    3. Объединение рангов алгоритмом Reciprocal Rank Fusion (RRF)
    """
    docs, metadatas, ids = _get_all_chroma_docs()
    if not docs:
        return []

    # --- 1. Векторный поиск (ChromaDB) ---
    vector_res = collection.query(query_texts=[query], n_results=min(top_k * 2, len(docs)))
    vector_ids = vector_res["ids"][0] if vector_res["ids"] else []

    # --- 2. Поиск по ключевым словам (BM25) ---
    tokenized_corpus = [doc.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    bm25_ids = [ids[i] for i in bm25_ranked_indices[:top_k * 2]]

    # --- 3. Reciprocal Rank Fusion (RRF) ---
    rrf_scores: Dict[str, float] = {}
    k = 60  # Константа сглаживания RRF
    id_map = {ids[i]: (docs[i], metadatas[i]) for i in range(len(ids))}

    for rank, doc_id in enumerate(vector_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

    for rank, doc_id in enumerate(bm25_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

    # Сортировка и отбор кандидатных документов
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

    candidates = []
    for doc_id in sorted_doc_ids:
        doc_text, meta = id_map[doc_id]
        candidates.append({
            "id": doc_id,
            "child_text": doc_text,
            "parent_text": meta.get("parent_text", doc_text),
            "meta": meta
        })

    return candidates


def rerank_passages(query: str, candidates: List[Dict[str, Any]], top_n: int = 2) -> List[str]:
    """Реранкинг кандидатов с помощью FlashRank и извлечение Parent-текста."""
    if not candidates:
        return []

    # Если FlashRank недоступен, возвращаем родительский контекст Top-N без реранкинга
    if not ranker:
        unique_parents = list(dict.fromkeys([c["parent_text"] for c in candidates]))
        return unique_parents[:top_n]

    passages = [
        {"id": idx, "text": c["parent_text"]}
        for idx, c in enumerate(candidates)
    ]

    rerank_request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_request)

    ranked_parent_texts = []
    for item in results[:top_n]:
        original_idx = item["id"]
        parent_text = candidates[original_idx]["parent_text"]
        if parent_text not in ranked_parent_texts:
            ranked_parent_texts.append(parent_text)

    return ranked_parent_texts


# =====================================================================
# 3. Основная задача Celery
# =====================================================================
@app.task(name="process_ai_task")
def process_ai_task(chat_id: int, user_message: str) -> Dict[str, Any]:
    # ШАГ A: Извлечение контекста через Advanced RAG
    candidates = hybrid_search(user_message, top_k=6)
    relevant_contexts = rerank_passages(user_message, candidates, top_n=5)

    context_str = "\n---\n".join(relevant_contexts) if relevant_contexts else "Контекст из базы знаний не найден."

    system_prompt = (
        "Ты — AI-консультант интернет-магазина.\n"
        "1. Отвечай на вопросы пользователя по каталогу и правилам магазина строго на основе КОНТЕКСТА.\n"
        "2. Если вопрос касается статуса или отмены заказа, ОБЯЗАТЕЛЬНО используй доступные инструменты (Tools).\n\n"
        f"КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:\n{context_str}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    # Используем бесплатную модель на OpenRouter
    model_name = "openrouter/free"

    try:
        # ШАГ B: Запрос к LLM со схемами инструментов
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto"
        )

        response_message = response.choices[0].message

        # ШАГ C: Проверка и обработка вызова функций (Function Calling)
        if response_message.tool_calls:
            messages.append(response_message)

            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                if func_name in AVAILABLE_TOOLS:
                    # Динамическое исполнение Python-функции
                    tool_output = AVAILABLE_TOOLS[func_name](**func_args)

                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": func_name,
                        "content": tool_output
                    })

            # ШАГ D: Финальный генеративный запрос к LLM с результатами работы функций
            final_response = client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            final_text = final_response.choices[0].message.content
        else:
            final_text = response_message.content

    except Exception as e:
        final_text = f"Произошла ошибка при обработке запроса: {str(e)}"

    return {"chat_id": chat_id, "response": final_text}
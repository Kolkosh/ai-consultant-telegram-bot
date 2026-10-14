import json
import os
import uuid
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Инициализируем локальную ChromaDB в папке ./chroma_db
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="store_knowledge_base")


def index_catalog():
    """Индексация товаров из catalog.json"""
    catalog_path = "data/catalog.json"
    if not os.path.exists(catalog_path):
        print(f"Ошибка: файл {catalog_path} не найден!")
        return

    with open(catalog_path, "r", encoding="utf-8") as f:
        products = json.load(f)

    docs, metadatas, ids = [], [], []
    for prod in products:
        text_representation = (
            f"Товар: {prod['name']}. Артикул: {prod['sku']}. "
            f"Категория: {prod['category']}. Цена: {prod['price']} руб. "
            f"Гарантия: {prod['warranty_months']} мес. Описание: {prod['description']}"
        )

        docs.append(text_representation)
        metadatas.append(
            {
                "sku": prod["sku"],
                "type": "product",
                "parent_text": text_representation,  # Для товаров родительский и дочерний текст совпадают
            }
        )
        ids.append(f"prod_{prod['sku']}")

    collection.add(documents=docs, metadatas=metadatas, ids=ids)
    print(f"Индексировано товаров: {len(products)}")


def index_rules():
    """Индексация регламентов из rules.txt с паттерном Parent-Child RAG"""
    rules_path = "data/rules.txt"
    if not os.path.exists(rules_path):
        print(f"Ошибка: файл {rules_path} не найден!")
        return

    with open(rules_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # 1. Создаем родительский сплиттер (крупные логические блоки для контекста нейросети)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    # 2. Создаем дочерний сплиттер (мелкие куски для точного векторного поиска)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=120, chunk_overlap=20
    )

    parent_chunks = parent_splitter.split_text(full_text)

    docs, metadatas, ids = [], [], []
    for parent_text in parent_chunks:
        parent_id = str(uuid.uuid4())
        children = child_splitter.split_text(parent_text)

        for child_text in children:
            docs.append(child_text)
            metadatas.append(
                {
                    "type": "rule",
                    "parent_id": parent_id,
                    "parent_text": parent_text,  # В метаданных сохраняем ПОЛНЫЙ родительский контекст
                }
            )
            ids.append(str(uuid.uuid4()))

    collection.add(documents=docs, metadatas=metadatas, ids=ids)
    print(
        f"Индексировано правил: {len(parent_chunks)} родительских блоков ({len(docs)} дочерних чанков)"
    )


if __name__ == "__main__":
    # Очищаем базу перед повторной индексацией
    try:
        existing_ids = collection.get()["ids"]
        if existing_ids:
            collection.delete(ids=existing_ids)
            print("Старая коллекция очищена.")
    except Exception as e:
        print(f"Предупреждение при очистке коллекции: {e}")

    index_catalog()
    index_rules()
    print("--- Индексация полностью завершена! ---")
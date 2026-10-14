import json
from typing import Dict, Callable
from pydantic import BaseModel, Field, ValidationError

# =====================================================================
# 1. Mock-База Данных (Имитация реальной СУБД / CRM магазина)
# =====================================================================
MOCK_ORDERS_DB = {
    "8921": {"status": "В сборке", "item": "Беспроводной пылесос Xiaomi Vacuum Cleaner G10"},
    "5541": {"status": "Передан в доставку", "item": "Ноутбук Apple MacBook Air 13 M2"},
    "1002": {"status": "Доставлен", "item": "Наушники AirPods Pro"}
}


# =====================================================================
# 2. Pydantic-схема для строгой валидации входящих параметров
# =====================================================================
class OrderIdSchema(BaseModel):
    order_id: str = Field(
        ...,
        description="Идентификатор заказа, состоящий из цифр",
        min_length=1
    )


# =====================================================================
# 3. Реализация функций-инструментов (Executable Tools)
# =====================================================================
def get_order_status(order_id: str) -> str:
    """Получает текущий статус заказа из базы данных."""
    try:
        validated = OrderIdSchema(order_id=str(order_id))
        oid = validated.order_id
    except ValidationError as e:
        return json.dumps({"error": "Неверный формат номера заказа", "details": str(e)}, ensure_ascii=False)

    order = MOCK_ORDERS_DB.get(oid)
    if not order:
        return json.dumps({"error": f"Заказ #{oid} не найден в базе данных."}, ensure_ascii=False)

    return json.dumps({
        "order_id": oid,
        "status": order["status"],
        "item": order["item"]
    }, ensure_ascii=False)


def cancel_order(order_id: str) -> str:
    """Отменяет заказ, если он находится в статусе 'В сборке'."""
    try:
        validated = OrderIdSchema(order_id=str(order_id))
        oid = validated.order_id
    except ValidationError as e:
        return json.dumps({"error": "Неверный формат номера заказа", "details": str(e)}, ensure_ascii=False)

    order = MOCK_ORDERS_DB.get(oid)
    if not order:
        return json.dumps({"error": f"Заказ #{oid} не найден в базе данных."}, ensure_ascii=False)

    if order["status"] == "В сборке":
        order["status"] = "Отменен"
        return json.dumps({
            "order_id": oid,
            "success": True,
            "message": f"Заказ #{oid} успешно отменен в системе."
        }, ensure_ascii=False)
    else:
        return json.dumps({
            "order_id": oid,
            "success": False,
            "current_status": order["status"],
            "reason": f"Нельзя отменить заказ в статусе '{order['status']}'. Отмена возможна только со статусом 'В сборке'."
        }, ensure_ascii=False)


# Реестр функций для динамического вызова в Celery-воркере
AVAILABLE_TOOLS: Dict[str, Callable[..., str]] = {
    "get_order_status": get_order_status,
    "cancel_order": cancel_order,
}


# =====================================================================
# 4. Спецификация инструментов (JSON Schema) для LLM
# =====================================================================
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Узнать статус заказа по его уникальному номеру/идентификатору.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Идентификатор заказа, например '8921'."
                    }
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Запросить отмену заказа по его номеру. Срабатывает только если заказ еще не передан в доставку (статус 'В сборке').",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Идентификатор заказа для отмены, например '8921'."
                    }
                },
                "required": ["order_id"]
            }
        }
    }
]
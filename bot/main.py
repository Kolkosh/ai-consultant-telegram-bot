import asyncio
import logging
import os
from aiogram import Bot, Dispatcher, types
from aiogram.enums import ChatAction
from aiogram.filters import CommandStart
from dotenv import load_dotenv

from worker.tasks import process_ai_task

load_dotenv()

logging.basicConfig(level=logging.INFO)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

@dp.message(CommandStart())
async def start_handler(message: types.Message):
    await message.answer(
        "Привет! Я AI-консультант интернет-магазина.\n\n"
        "Я могу:\n"
        "• Ответить на вопросы по товарам и правилам магазина\n"
        "• Проверить статус заказа (например: 'Где мой заказ 8921?')\n"
        "• Отменить заказ (например: 'Отмени заказ 8921')"
    )


@dp.message()
async def handle_user_message(message: types.Message):
    # Показываем статус "печатает...", пока Celery обрабатывает запрос
    await bot.send_chat_action(
        chat_id=message.chat.id, action=ChatAction.TYPING
    )

    # 1. Отправляем задачу в очередь Celery (не блокируя бот)
    task = process_ai_task.delay(message.chat.id, message.text)

    # 2. Неблокирующее ожидание выполнения задачи в Celery
    while not task.ready():
        await asyncio.sleep(0.4)

    # 3. Извлекаем результат из Redis
    result = task.result
    response_text = result.get(
        "response", "Произошла ошибка при получении ответа."
    )

    await message.answer(response_text)


async def main():
    logging.info("Запуск Telegram бота...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
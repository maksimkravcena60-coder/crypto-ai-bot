import os
import json
import time
import requests
from typing import Optional, Dict, Any, List, Tuple

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

from apscheduler.schedulers.background import BackgroundScheduler
from openai import OpenAI


# =========================
# ENV
# =========================
load_dotenv()
TELEGRAM_TOKEN = (os.getenv("TELEGRAM_TOKEN") or "").strip()
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
CRYPTOPANIC_API_KEY = (os.getenv("CRYPTOPANIC_API_KEY") or "").strip()

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN пустой. Добавь токен в .env")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY пустой. Добавь ключ в .env")

client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# Storage: chat_ids + watchlist (простая)
# =========================
DATA_DIR = "data"
STATE_FILE = os.path.join(DATA_DIR, "state.json")

DEFAULT_STATE = {
    "chat_ids": [],
    "watchlist": ["bitcoin", "ethereum", "solana"],  # можешь менять
}

def ensure_state():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(STATE_FILE):
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_STATE, f, ensure_ascii=False, indent=2)

def load_state() -> Dict[str, Any]:
    ensure_state()
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_state(state: Dict[str, Any]):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def add_chat_id(chat_id: int):
    state = load_state()
    ids = set(int(x) for x in state.get("chat_ids", []))
    if chat_id not in ids:
        ids.add(chat_id)
        state["chat_ids"] = sorted(ids)
        save_state(state)

def get_chat_ids() -> List[int]:
    state = load_state()
    return [int(x) for x in state.get("chat_ids", [])]

def get_watchlist() -> List[str]:
    state = load_state()
    wl = state.get("watchlist", [])
    return wl if isinstance(wl, list) and wl else ["bitcoin", "ethereum", "solana"]


# =========================
# HTTP helpers
# =========================
def http_get_json(url: str, params: Optional[dict] = None, timeout: int = 25) -> Dict[str, Any]:
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


# =========================
# CoinGecko helpers
# =========================
def cg_price(ids: List[str], vs: str = "usd") -> Dict[str, float]:
    data = http_get_json(
        "https://api.coingecko.com/api/v3/simple/price",
        params={"ids": ",".join(ids), "vs_currencies": vs},
        timeout=20
    )
    out = {}
    for cid in ids:
        if cid in data and vs in data[cid]:
            out[cid] = float(data[cid][vs])
    return out

def get_btc_dominance_pct() -> Optional[float]:
    try:
        data = http_get_json("https://api.coingecko.com/api/v3/global", timeout=20)
        return float(data["data"]["market_cap_percentage"]["btc"])
    except Exception:
        return None


# =========================
# CryptoPanic news
# =========================
def get_top_news(limit: int = 6) -> List[Dict[str, str]]:
    if not CRYPTOPANIC_API_KEY:
        return []
    url = "https://cryptopanic.com/api/v1/posts/"
    params = {
        "auth_token": CRYPTOPANIC_API_KEY,
        "public": "true",
        "kind": "news",
        "filter": "rising",
    }
    data = http_get_json(url, params=params, timeout=30)
    out = []
    for item in data.get("results", [])[:limit]:
        title = item.get("title") or ""
        link = item.get("url") or item.get("original_url") or ""
        if title:
            out.append({"title": title, "url": link})
    return out


# =========================
# Liquidations (MVP link + explanation)
# =========================
def liquidation_link(symbol: str = "btc") -> str:
    # Для картинки нужен спец-источник. Сегодня: ссылка + объяснение.
    return "https://www.coinglass.com/pro/futures/LiquidationHeatMap"


# =========================
# AI Intent + parsing symbols
# =========================
INTENT_SYSTEM = """
Ты крипто-ассистент в Telegram. Пользователь пишет как человек.
Нужно понять, что он хочет, даже если фраза разговорная.

Верни ТОЛЬКО JSON.

intent:
- market_overview  (сводка/обзор рынка: "что по рынку", "как рынок", "привет", "что происходит", "дай сводку", "как дела")
- prices           (цены монет: "сколько btc", "цена эфира", "btc eth sol", "что с соланой")
- dominance        (доминация BTC)
- news             (новости/что важного)
- liquidations     (ликвидации/heatmap/карта ликвидаций)
- help             (если не понял)

coins:
- массив тикеров, если пользователь явно просит монеты (например ["BTC","ETH","SOL"])
- если не указал — пустой массив []

Важное правило:
Если сообщение похоже на общий вопрос/приветствие, считай это market_overview.

Формат:
{"intent":"...", "coins":["BTC","ETH"]}
"""

def detect_intent(user_text: str) -> Dict[str, Any]:
    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": INTENT_SYSTEM},
            {"role": "user", "content": user_text},
        ],
        temperature=0.0,
    )
    content = (r.choices[0].message.content or "").strip()
    try:
        obj = json.loads(content)
        intent = obj.get("intent", "help")
        coins = obj.get("coins", [])
        if not isinstance(coins, list):
            coins = []
        coins = [str(c).upper().strip() for c in coins if str(c).strip()]
        return {"intent": intent, "coins": coins}
    except Exception:
        return {"intent": "help", "coins": []}


# =========================
# Coin mapping (тикер -> coingecko id)
# =========================
COIN_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "TON": "the-open-network",
    "TRX": "tron",
    "AVAX": "avalanche-2",
    "MATIC": "polygon-pos",
    "DOT": "polkadot",
    "LTC": "litecoin",
    "LINK": "chainlink",
}

def normalize_coins(requested: List[str]) -> List[str]:
    # Если не указал — используем watchlist
    if not requested:
        return ["BTC", "ETH", "SOL"]
    # Оставляем только те, которые знаем
    out = []
    for t in requested:
        if t in COIN_MAP:
            out.append(t)
    # если все неизвестны — fallback к базовым
    return out if out else ["BTC", "ETH", "SOL"]

def prices_text(coins: List[str]) -> str:
    ids = [COIN_MAP[c] for c in coins]
    prices = cg_price(ids, "usd")
    lines = []
    for c in coins:
        cid = COIN_MAP[c]
        if cid in prices:
            lines.append(f"{c}: ${prices[cid]:,.4f}" if prices[cid] < 1 else f"{c}: ${prices[cid]:,.2f}")
        else:
            lines.append(f"{c}: —")
    return "\n".join(lines)


# =========================
# AI content generation
# =========================
def ai_market_brief(coins: List[str]) -> str:
    ids = [COIN_MAP[c] for c in coins]
    p = cg_price(ids, "usd")
    dom = get_btc_dominance_pct()
    news = get_top_news(6)
    liq = liquidation_link("btc")

    coin_lines = []
    for c in coins:
        cid = COIN_MAP[c]
        val = p.get(cid)
        if val is None:
            coin_lines.append(f"{c}: —")
        else:
            coin_lines.append(f"{c}: ${val:,.4f}" if val < 1 else f"{c}: ${val:,.2f}")

    news_block = "\n".join([f"- {n['title']}\n  {n.get('url','')}".strip() for n in news]) if news else "— (новости не подключены или нет результата)"
    dom_line = f"{dom:.2f}%" if dom is not None else "—"

    prompt = f"""
Сделай короткий, практичный обзор рынка (до 14 строк). Язык: русский.
Данные:
Цены:
{chr(10).join(coin_lines)}
BTC доминация: {dom_line}
Новости:
{news_block}
Ликвидации (heatmap): {liq}

Формат:
1) 2-4 строки: что по рынку сейчас
2) Что важно сегодня (2-4 пункта)
3) Риски/триггеры (2-4 пункта)
4) 2-3 строки: как читать heatmap ликвидаций (где кластеры, что значит магнит)
"""
    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return (r.choices[0].message.content or "").strip()

def ai_liquidations_explain(symbol: str) -> str:
    liq = liquidation_link(symbol)
    prompt = f"""
Поясни ликвидации/heatmap для {symbol}.
Коротко и понятно (8-12 строк):
- что такое кластеры ликвидаций
- как использовать уровни
- что значит "вынос"
- риск-заметка
В конце ссылка: {liq}
"""
    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return (r.choices[0].message.content or "").strip()

def ai_news_digest(news: List[Dict[str, str]]) -> str:
    if not news:
        return "Новости не подключены или нет результата."
    news_block = "\n".join([f"- {n['title']}\n  {n.get('url','')}".strip() for n in news])
    prompt = f"""
Вот новости:
{news_block}

Сделай вывод:
- какие 1-2 новости high impact для крипты и почему
- что мониторить сегодня (1-2 пункта)
Коротко.
"""
    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return (r.choices[0].message.content or "").strip()

def ai_is_high_impact_news(title: str) -> bool:
    prompt = f"""
Заголовок: "{title}"
Это high-impact для крипторынка в ближайшие 24-72 часа? (регуляции, войны, ставки, ETF, санкции, биржи, взломы)
Ответь строго: YES или NO.
"""
    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    ans = (r.choices[0].message.content or "").strip().upper()
    return ans.startswith("YES")


# =========================
# Telegram handlers
# =========================
HELP_TEXT = (
    "Пиши как угодно, я пойму 🙂\n\n"
    "Примеры:\n"
    "• «привет, что по рынку?»\n"
    "• «как там рынок?»\n"
    "• «цена btc и eth»\n"
    "• «доминация биткоина»\n"
    "• «новости»\n"
    "• «карта ликвидаций btc»\n"
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    add_chat_id(update.effective_chat.id)
    await update.message.reply_text("Я запущен ✅\n\n" + HELP_TEXT)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    add_chat_id(update.effective_chat.id)
    user_text = (update.message.text or "").strip()

    intent = detect_intent(user_text)
    kind = intent["intent"]
    coins = normalize_coins(intent.get("coins", []))

    try:
        if kind == "market_overview":
            brief = ai_market_brief(coins)
            # добавим компактные цифры наверху
            dom = get_btc_dominance_pct()
            header = "📊 Сводка рынка\n\n" + prices_text(coins)
            header += f"\n\nBTC доминация: {dom:.2f}%\n\n" if dom is not None else "\n\nBTC доминация: —\n\n"
            await update.message.reply_text(header + brief)
            return

        if kind == "prices":
            await update.message.reply_text("💰 Цены:\n" + prices_text(coins))
            return

        if kind == "dominance":
            dom = get_btc_dominance_pct()
            await update.message.reply_text(f"BTC доминация: {dom:.2f}%" if dom is not None else "Не удалось получить доминацию сейчас.")
            return

        if kind == "news":
            news = get_top_news(6)
            if not news:
                await update.message.reply_text("Новости не подключены или нет результата (проверь CRYPTOPANIC_API_KEY).")
                return
            txt = "📰 Топ новости:\n\n" + "\n\n".join([f"- {n['title']}\n{n.get('url','')}".strip() for n in news])
            digest = ai_news_digest(news)
            await update.message.reply_text(txt + "\n\n" + digest)
            return

        if kind == "liquidations":
            # для простоты берём первую монету из списка
            sym = coins[0] if coins else "BTC"
            msg = ai_liquidations_explain(sym)
            await update.message.reply_text(msg)
            return

        # help / fallback: отвечаем умно на любой текст
        # + добавим небольшой market context
        ids = [COIN_MAP[c] for c in coins]
        p = cg_price(ids, "usd")
        dom = get_btc_dominance_pct()
        news = get_top_news(4)
        news_titles = "\n".join([f"- {n['title']}" for n in news]) if news else "—"
        dom_line = f"{dom:.2f}%" if dom is not None else "—"

        ctx_prices = []
        for c in coins:
            val = p.get(COIN_MAP[c])
            ctx_prices.append(f"{c}={val}" if val is not None else f"{c}=—")

        prompt = f"""
Пользователь: {user_text}

Контекст рынка:
Цены: {", ".join(ctx_prices)}
BTC доминация: {dom_line}
Новости: {news_titles}

Ответь как умный крипто-помощник: сначала ответь на пользователя, затем 2-4 строки что по рынку.
"""
        r = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        await update.message.reply_text((r.choices[0].message.content or "").strip())

    except Exception as e:
        await update.message.reply_text(f"Ошибка: {type(e).__name__}: {e}")


# =========================
# AUTO ALERTS (volatility + high-impact news)
# =========================
_last_prices: Optional[Dict[str, float]] = None
_last_news_titles: set[str] = set()
_last_alert_ts: float = 0.0

def _rate_limit(seconds: int) -> bool:
    global _last_alert_ts
    now = time.time()
    if now - _last_alert_ts < seconds:
        return True
    _last_alert_ts = now
    return False

def alert_job(app: Application):
    """
    Every 5 minutes:
    - volatility alert for BTC/ETH/SOL (watchlist)
    - high impact news alert (filtered by AI)
    """
    global _last_prices, _last_news_titles

    chat_ids = get_chat_ids()
    if not chat_ids:
        return

    coins = ["BTC", "ETH", "SOL"]  # базовый набор для алертов
    ids = [COIN_MAP[c] for c in coins]

    # 1) Volatility alerts
    try:
        current = cg_price(ids, "usd")
        # current: {"bitcoin": price, ...}
        if _last_prices is not None:
            for c in coins:
                cid = COIN_MAP[c]
                if cid in current and cid in _last_prices and _last_prices[cid] > 0:
                    change = abs(current[cid] - _last_prices[cid]) / _last_prices[cid] * 100.0
                    if change >= 2.0 and not _rate_limit(180):  # антиспам общий
                        msg = f"🚨 Волатильность {c}\nИзменение: {change:.2f}%\nЦена: ${current[cid]:,.2f}"
                        for chat_id in chat_ids:
                            app.bot.send_message(chat_id, msg)
        _last_prices = current
    except Exception:
        pass

    # 2) High-impact news alerts
    if CRYPTOPANIC_API_KEY:
        try:
            news = get_top_news(6)
            fresh = [n for n in news if n.get("title") and n["title"] not in _last_news_titles]
            if fresh and not _rate_limit(180):
                # фильтруем через AI
                sent = 0
                for n in fresh:
                    title = n["title"]
                    if ai_is_high_impact_news(title):
                        msg = "📰 High-impact новость:\n\n" + title
                        if n.get("url"):
                            msg += "\n" + n["url"]
                        for chat_id in chat_ids:
                            app.bot.send_message(chat_id, msg)
                        sent += 1
                        if sent >= 2:
                            break
            _last_news_titles = {n["title"] for n in news if n.get("title")}
        except Exception:
            pass


# =========================
# MAIN
# =========================
def main():
    ensure_state()

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    scheduler = BackgroundScheduler()
    scheduler.add_job(alert_job, "interval", minutes=5, args=[app])
    scheduler.start()

    print("Bot running... (Ctrl+C to stop)")
    app.run_polling()

if __name__ == "__main__":
    main()
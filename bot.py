import os
import json
import time
import requests
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

from apscheduler.schedulers.background import BackgroundScheduler
from openai import OpenAI

import feedparser


# =========================
# ENV
# =========================
load_dotenv()
TELEGRAM_TOKEN = (os.getenv("TELEGRAM_TOKEN") or "").strip()
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN пустой. Добавь токен в .env")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY пустой. Добавь ключ в .env")

client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# Storage: chat ids
# =========================
DATA_DIR = "data"
STATE_FILE = os.path.join(DATA_DIR, "state.json")

def ensure_state():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(STATE_FILE):
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"chat_ids": []}, f, ensure_ascii=False, indent=2)

def load_chat_ids() -> List[int]:
    ensure_state()
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list({int(x) for x in data.get("chat_ids", [])})

def add_chat_id(chat_id: int):
    ids = load_chat_ids()
    if chat_id not in ids:
        ids.append(chat_id)
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"chat_ids": ids}, f, ensure_ascii=False, indent=2)


# =========================
# HTTP helpers
# =========================
def http_get_json(url: str, params: Optional[dict] = None, timeout: int = 25) -> Dict[str, Any]:
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def safe_get_json(url: str, params: Optional[dict] = None, timeout: int = 25) -> Optional[Dict[str, Any]]:
    try:
        return http_get_json(url, params=params, timeout=timeout)
    except Exception:
        return None


# =========================
# Market (CoinGecko)
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
    "DOT": "polkadot",
    "LTC": "litecoin",
    "LINK": "chainlink",
}

def cg_prices(symbols: List[str]) -> Dict[str, float]:
    ids = [COIN_MAP[s] for s in symbols if s in COIN_MAP]
    if not ids:
        ids = ["bitcoin"]
    data = http_get_json(
        "https://api.coingecko.com/api/v3/simple/price",
        params={"ids": ",".join(ids), "vs_currencies": "usd"},
        timeout=20,
    )
    out = {}
    for sym in symbols:
        cid = COIN_MAP.get(sym)
        if cid and cid in data and "usd" in data[cid]:
            out[sym] = float(data[cid]["usd"])
    return out

def get_btc_dominance_pct() -> Optional[float]:
    data = safe_get_json("https://api.coingecko.com/api/v3/global", timeout=20)
    if not data:
        return None
    try:
        return float(data["data"]["market_cap_percentage"]["btc"])
    except Exception:
        return None

def fmt_price(v: float) -> str:
    return f"${v:,.4f}" if v < 1 else f"${v:,.2f}"


# =========================
# RSS news (без ключей)
# =========================
RSS_FEEDS = [
    # Можно менять/добавлять
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
]

def fetch_rss_news(limit: int = 10) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:limit]:
                title = (e.get("title") or "").strip()
                link = (e.get("link") or "").strip()
                if title:
                    items.append({"title": title, "url": link})
        except Exception:
            continue

    # убираем дубли по title
    seen = set()
    uniq = []
    for n in items:
        t = n["title"]
        if t not in seen:
            seen.add(t)
            uniq.append(n)
    return uniq[:limit]


# =========================
# Liquidations
# =========================
def liquidation_link() -> str:
    return "https://www.coinglass.com/pro/futures/LiquidationHeatMap"


# =========================
# AI intent
# =========================
INTENT_SYSTEM = """
Ты крипто-ассистент в Telegram. Пользователь пишет как человек.
Определи intent и монеты. Верни только JSON.

intent:
- market_overview (обзор/сводка/привет/что по рынку/как рынок)
- prices (цены монет)
- dominance (доминация BTC)
- news (новости)
- liquidations (ликвидации/heatmap)
- help (если не понял)

coins: массив тикеров (BTC,ETH,SOL...) если пользователь явно упомянул, иначе [].
Если сообщение выглядит как общий вопрос/привет — market_overview.

Формат: {"intent":"...", "coins":["BTC","ETH"]}
"""

def detect_intent(text: str) -> Dict[str, Any]:
    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "system", "content": INTENT_SYSTEM},
                  {"role": "user", "content": text}],
        temperature=0.0,
    )
    content = (r.choices[0].message.content or "").strip()
    try:
        obj = json.loads(content)
        coins = obj.get("coins", [])
        if not isinstance(coins, list):
            coins = []
        coins = [str(c).upper().strip() for c in coins if str(c).strip()]
        return {"intent": obj.get("intent", "help"), "coins": coins}
    except Exception:
        return {"intent": "help", "coins": []}

def normalize_coins(coins: List[str]) -> List[str]:
    if not coins:
        return ["BTC", "ETH", "SOL"]
    known = [c for c in coins if c in COIN_MAP]
    return known if known else ["BTC", "ETH", "SOL"]


# =========================
# AI helpers (summary + фильтр важности)
# =========================
def ai_news_digest(news: List[Dict[str, str]]) -> str:
    if not news:
        return "Новостей сейчас не нашёл (RSS временно недоступен)."
    block = "\n".join([f"- {n['title']}\n  {n.get('url','')}".strip() for n in news[:6]])
    prompt = f"""
Вот новости:
{block}

Сделай коротко:
- какие 1-2 новости high-impact для крипты и почему
- что мониторить сегодня
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

def ai_market_brief(coins: List[str]) -> str:
    prices = cg_prices(coins)
    dom = get_btc_dominance_pct()
    news = fetch_rss_news(8)
    liq = liquidation_link()

    price_lines = "\n".join([f"{c}: {fmt_price(prices[c])}" for c in coins if c in prices])
    dom_line = f"{dom:.2f}%" if dom is not None else "—"
    news_block = "\n".join([f"- {n['title']}" for n in news[:6]]) if news else "— (RSS недоступен)"

    prompt = f"""
Сделай короткий, практичный обзор рынка (до 14 строк), русский.
Данные:
Цены:
{price_lines}
BTC доминация: {dom_line}
Новости (RSS):
{news_block}
Ликвидации: {liq}

Формат:
1) 2-4 строки что по рынку
2) Что важно сегодня (2-4 пункта)
3) Риски/триггеры (2-4 пункта)
4) 2 строки как читать heatmap ликвидаций
"""
    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return (r.choices[0].message.content or "").strip()

def ai_liq_explain() -> str:
    liq = liquidation_link()
    prompt = f"""
Объясни простыми словами, что такое heatmap ликвидаций и как её читать (8-12 строк).
В конце дай ссылку: {liq}
"""
    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return (r.choices[0].message.content or "").strip()


# =========================
# Telegram handlers
# =========================
HELP = (
    "Пиши как угодно 🙂\n"
    "Примеры:\n"
    "• привет, что по рынку?\n"
    "• цены btc eth sol\n"
    "• доминация\n"
    "• новости\n"
    "• ликвидации\n"
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    add_chat_id(update.effective_chat.id)
    await update.message.reply_text("Я запущен ✅\n\n" + HELP)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP)

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    add_chat_id(update.effective_chat.id)
    text = (update.message.text or "").strip()

    intent = detect_intent(text)
    kind = intent.get("intent", "help")
    coins = normalize_coins(intent.get("coins", []))

    if kind == "prices":
        p = cg_prices(coins)
        lines = [f"{c}: {fmt_price(p[c])}" for c in coins if c in p]
        await update.message.reply_text("💰 Цены:\n" + ("\n".join(lines) if lines else "Не удалось получить цены сейчас."))
        return

    if kind == "dominance":
        dom = get_btc_dominance_pct()
        await update.message.reply_text(f"BTC доминация: {dom:.2f}%" if dom is not None else "Не удалось получить доминацию сейчас.")
        return

    if kind == "news":
        news = fetch_rss_news(8)
        if not news:
            await update.message.reply_text("RSS новости сейчас недоступны. Попробуй позже.")
            return
        digest = ai_news_digest(news)
        txt = "📰 RSS Новости:\n\n" + "\n\n".join([f"- {n['title']}\n{n.get('url','')}".strip() for n in news[:6]])
        await update.message.reply_text(txt + "\n\n" + digest)
        return

    if kind == "liquidations":
        await update.message.reply_text(ai_liq_explain())
        return

    if kind == "market_overview":
        brief = ai_market_brief(coins)
        dom = get_btc_dominance_pct()
        p = cg_prices(coins)
        header_lines = [f"{c}: {fmt_price(p[c])}" for c in coins if c in p]
        header = "📊 Сводка рынка\n" + ("\n".join(header_lines) if header_lines else "")
        header += f"\nBTC доминация: {dom:.2f}%\n\n" if dom is not None else "\nBTC доминация: —\n\n"
        await update.message.reply_text(header + brief)
        return

    await update.message.reply_text("Не до конца понял. " + HELP)


# =========================
# Alerts (волатильность + важные RSS новости)
# =========================
_last_prices: Optional[Dict[str, float]] = None
_last_news_titles: set[str] = set()
_last_alert_ts: float = 0.0

def rate_limit(seconds: int) -> bool:
    global _last_alert_ts
    now = time.time()
    if now - _last_alert_ts < seconds:
        return True
    _last_alert_ts = now
    return False

def alert_job(app: Application):
    global _last_prices, _last_news_titles

    chat_ids = load_chat_ids()
    if not chat_ids:
        return

    # Volatility: BTC/ETH/SOL
    coins = ["BTC", "ETH", "SOL"]
    try:
        current = cg_prices(coins)  # {BTC:..., ETH:...}
        if _last_prices is not None:
            for c in coins:
                if c in current and c in _last_prices and _last_prices[c] > 0:
                    change = abs(current[c] - _last_prices[c]) / _last_prices[c] * 100
                    if change >= 2.0 and not rate_limit(180):
                        msg = f"🚨 Волатильность {c}\nИзменение: {change:.2f}%\nЦена: {fmt_price(current[c])}"
                        for cid in chat_ids:
                            app.bot.send_message(cid, msg)
        _last_prices = current
    except Exception:
        pass

    # High-impact RSS news
    try:
        news = fetch_rss_news(10)
        fresh = [n for n in news if n.get("title") and n["title"] not in _last_news_titles]
        if fresh and not rate_limit(180):
            sent = 0
            for n in fresh:
                title = n["title"]
                if ai_is_high_impact_news(title):
                    msg = "📰 High-impact новость (RSS):\n\n" + title
                    if n.get("url"):
                        msg += "\n" + n["url"]
                    for cid in chat_ids:
                        app.bot.send_message(cid, msg)
                    sent += 1
                    if sent >= 2:
                        break
        _last_news_titles = {n["title"] for n in news if n.get("title")}
    except Exception:
        pass


# =========================
# Main
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

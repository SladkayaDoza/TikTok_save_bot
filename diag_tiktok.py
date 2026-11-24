"""Диагностика: почему CDN TikTok не отдаёт видео своему парсеру.

Запуск на сервере, из папки бота:
    python diag_tiktok.py              # тестовый ролик
    python diag_tiktok.py 7710219959191751958

Ничего не меняет, только печатает: внешний IP, куда резолвятся хосты CDN,
что отдаёт страница и как CDN отвечает на разные варианты запроса.
"""
import asyncio
import json
import re
import shutil
import socket
import subprocess
import sys

import aiohttp

VIDEO_ID = sys.argv[1] if len(sys.argv) > 1 else "7709157631360847117"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0 Safari/537.36")
HEADERS = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}
# Узел Akamai, через который у разработчика всё работает (для сравнения)
REFERENCE_EDGE = "184.24.77.49"


def resolve(host: str) -> dict:
    out = {}
    for fam, name in ((socket.AF_INET, "v4"), (socket.AF_INET6, "v6")):
        try:
            out[name] = sorted({ai[4][0] for ai in socket.getaddrinfo(host, 443, fam)})[:3]
        except Exception as e:
            out[name] = f"нет ({type(e).__name__})"
    return out


def curl_probe(url: str, cookie: str, resolve_ip: str | None = None) -> str:
    if not shutil.which("curl"):
        return "curl не установлен"
    host = url.split("/")[2]
    cmd = ["curl", "-s", "-4", "-m", "15", "-o", "/dev/null", "-r", "0-1023",
           "-w", "%{http_code} %{content_type} ip=%{remote_ip}",
           "-A", UA, "-H", f"Cookie: {cookie}", "-H", "Referer: https://www.tiktok.com/", url]
    if resolve_ip:
        cmd[1:1] = ["--resolve", f"{host}:443:{resolve_ip}"]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:
        return f"ошибка: {e}"


async def probe(session, url: str, headers: dict) -> str:
    try:
        async with session.get(url, headers=headers, allow_redirects=False,
                               timeout=aiohttp.ClientTimeout(total=15)) as r:
            peer = r.connection.transport.get_extra_info("peername") if r.connection else None
            body = await r.content.read(16)
            return (f"{r.status} {r.headers.get('Content-Type', '?')} "
                    f"ip={peer[0] if peer else '?'} начало={body[:12]!r}")
    except Exception as e:
        return f"ошибка: {type(e).__name__}: {e}"


async def main():
    print(f"Python {sys.version.split()[0]}, aiohttp {aiohttp.__version__}")

    async with aiohttp.ClientSession() as s:
        try:
            async with s.get("https://ipinfo.io/json", timeout=aiohttp.ClientTimeout(total=8)) as r:
                info = await r.json()
            print(f"Внешний IP: {info.get('ip')} ({info.get('country')}, {info.get('org')})")
        except Exception as e:
            print(f"Внешний IP: не узнать ({e})")

    for host in ("www.tiktok.com", "v16-webapp-prime.tiktok.com", "v19-webapp-prime.tiktok.com"):
        print(f"DNS {host}: {resolve(host)}")

    connector = aiohttp.TCPConnector(family=socket.AF_INET)
    async with aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar(), connector=connector) as s:
        page_url = f"https://www.tiktok.com/@_/video/{VIDEO_ID}"
        async with s.get(page_url, headers=HEADERS) as r:
            print(f"\nСтраница: HTTP {r.status}")
            page = await r.text()
        m = re.search(r'<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__"[^>]*>(.*?)</script>', page, re.S)
        if not m:
            print("В странице нет данных (капча/блок). Начало:", page[:200].replace("\n", " "))
            return
        scope = json.loads(m.group(1)).get("__DEFAULT_SCOPE__", {})
        ctx = scope.get("webapp.app-context") or {}
        print(f"Регион по мнению TikTok: {ctx.get('region')}, кластер: {ctx.get('clusterRegion')}")
        detail = scope.get("webapp.video-detail") or {}
        print(f"statusCode ролика: {detail.get('statusCode')}")
        video = (((detail.get("itemInfo") or {}).get("itemStruct") or {}).get("video")) or {}
        urls = []
        for b in video.get("bitrateInfo") or []:
            urls += (b.get("PlayAddr") or {}).get("UrlList") or []
        urls = list(dict.fromkeys(u for u in urls if u))[:2]
        if not urls:
            print("Ссылок на видео нет")
            return

        cookies = {c.key: c.value for c in s.cookie_jar}
        print(f"Куки: {sorted(cookies)}")
        cookie_header = "; ".join(f"{k}={v}" for k, v in cookies.items())
        base = {**HEADERS, "Referer": "https://www.tiktok.com/"}

        for u in urls:
            host = u.split("/")[2]
            print(f"\nCDN {host}:")
            print("  aiohttp, как бот (Range):  ", await probe(s, u, {**base, "Range": "bytes=0-1023"}))
            print("  aiohttp, без Range:        ", await probe(s, u, base))
            print("  aiohttp, без Referer:      ", await probe(s, u, {**HEADERS, "Range": "bytes=0-1023"}))
            print("  curl, DNS сервера:         ", curl_probe(u, cookie_header))
            print(f"  curl, узел {REFERENCE_EDGE}:", curl_probe(u, cookie_header, REFERENCE_EDGE))


asyncio.run(main())

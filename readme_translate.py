#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import re
import json
import time
import requests
import hashlib
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# ⚙️ 配置
API_KEY = "你的GROQ_API_KEY"
MODEL = "llama-3.3-70b-versatile"
API_URL = "https://api.groq.com/openai/v1/chat/completions"

CACHE_FILE = "translate_cache.json"
MAX_WORKERS = 5
MAX_LEN = 1200

# 🌏 进度控制
TOTAL_TASKS = 0
DONE_TASKS = 0
LOCK = Lock()

# 📦 缓存
def load_cache():
    if os.path.exists(CACHE_FILE):
        return json.load(open(CACHE_FILE, "r", encoding="utf-8"))
    return {}

def save_cache(cache):
    json.dump(cache, open(CACHE_FILE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

def get_key(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

CACHE = load_cache()

# 📥 输入
def get_input_content(input_arg):
    if os.path.exists(input_arg):
        with open(input_arg, "r", encoding="utf-8") as f:
            return f.read(), input_arg

    if input_arg.startswith("http"):
        url = input_arg
        if "github.com" in url and not url.endswith(".md"):
            url = url.replace("github.com", "raw.githubusercontent.com")
            url = url.replace("/blob/", "/")

        resp = requests.get(url)
        return resp.text, "output.md"

    return input_arg, "output.txt"

# Markdown保护
def split_markdown(md):
    return re.split(r"(```.*?```)", md, flags=re.S)

# 分块
def split_text(text):
    paragraphs = text.split("\n\n")
    chunks, current = [], ""

    for p in paragraphs:
        if len(current) + len(p) < MAX_LEN:
            current += p + "\n\n"
        else:
            chunks.append(current)
            current = p + "\n\n"

    if current:
        chunks.append(current)

    return chunks

# 统计任务
def count_total_chunks(content):
    total = 0
    for part in split_markdown(content):
        if not part.startswith("```"):
            total += len(split_text(part))
    return total

# 语言检测
def detect_language(text):
    zh = len(re.findall(r'[\u4e00-\u9fff]', text))
    return "zh" if zh / max(len(text), 1) > 0.3 else "other"

# API调用
def real_call_model(text, prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt + text}]
    }

    resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    return resp.json()["choices"][0]["message"]["content"]

# 进度
def print_progress():
    percent = DONE_TASKS / TOTAL_TASKS * 100 if TOTAL_TASKS else 100
    print(f"\r🌏 进度: {DONE_TASKS}/{TOTAL_TASKS} ({percent:.1f}%)", end="")

# 调用封装
def call_model(text, prompt):
    global DONE_TASKS

    key = get_key(text + prompt)

    if key in CACHE:
        with LOCK:
            DONE_TASKS += 1
            print_progress()
        return CACHE[key]

    try:
        result = real_call_model(text, prompt)
        CACHE[key] = result
    except:
        result = text

    with LOCK:
        DONE_TASKS += 1
        print_progress()

    return result

# 并发
def translate_chunks(chunks, prompt):
    results = [None] * len(chunks)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(call_model, c, prompt): i
            for i, c in enumerate(chunks)
        }

        for f in futures:
            results[futures[f]] = f.result()

    return results

# 翻译
def translate_text(text):
    lang = detect_language(text)

    if lang == "zh":
        prompt = "Translate into English, keep format:\n"
    else:
        prompt = (
            "翻译成简体中文：\n"
            "必须简体中文，不要解释，保留格式\n"
        )

    chunks = split_text(text)
    results = translate_chunks(chunks, prompt)

    return "".join(results)

def translate_content(content):
    result = []
    for part in split_markdown(content):
        if part.startswith("```"):
            result.append(part)
        else:
            result.append(translate_text(part))
    return "".join(result)

# 总结
def summarize(text):
    prompt = "用中文总结下面内容（200字以内）：\n"
    return real_call_model(text[:2000], prompt)

# 保存
def save_output(text, name):
    base, ext = os.path.splitext(name)
    if not ext:
        ext = ".txt"

    out = os.path.normpath(base + "_translated" + ext)

    with open(out, "w", encoding="utf-8") as f:
        f.write(text)

    save_cache(CACHE)

    print(f"\n\n✅ 输出: {out}")

def main():
    global TOTAL_TASKS, DONE_TASKS

    print(f"🤖 当前模型: {MODEL}")

    if not API_KEY:
        print("⚠️ 未配置 API_KEY，可能只使用缓存")

    input_arg = sys.argv[1]

    print("📥 读取内容...")
    content, name = get_input_content(input_arg)

    print("🔍 统计任务...")
    TOTAL_TASKS = count_total_chunks(content)
    DONE_TASKS = 0

    print(f"📦 总任务数: {TOTAL_TASKS}")
    print("🌏 开始翻译...")

    translated = translate_content(content)

    print("\n🧠 正在生成总结...")
    summary = summarize(translated)

    print("\n📌 内容总结：")
    print(summary)

    save_output(translated, name)

if __name__ == "__main__":
    main()
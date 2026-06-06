# 🐈 FatCat GPT API

This is an LLM inference server built using FastAPI and the `fatcat-gpt` package.
The model is configured as a Discord-active user named “Mimi,” who speaks bluntly and specializes in delivering brutally honest, emotionally damaging responses.

With this project, the model is loaded onto the GPU only once at server startup, and then stays hot in memory. It can continuously accept API requests and respond instantly, making it suitable for integration with Discord bots.

---

## 🛠️ Environment & Dependencies

Make sure you have installed the `fatcat-gpt` core package and all required dependencies:

```bash
# Ensure fatcat-gpt is installed
pip install fatcat-gpt
````

---

## 🚀 Quick Start Guide

# FatCat Project

English README — This document explains the project purpose, structure, installation, usage, and suggestions for development and testing.

---

## Brief Summary

This project provides a simple toolkit for deploying a custom “FatCat-style” language model as a local HTTP inference server, along with a client testing script and optional fine-tuning workflow.

Main features include:

* Loading a model on a GPU machine and exposing it via an API
* Sending test requests quickly
* Optional fine-tuning scripts

---

## Directory Structure & Key Files

* `fatcat_gpt/`

  * `core.py`: Encapsulates model logic (initialization, generation, utility functions)
  * `fatcat_phrases.json`: Phrase / response templates used for prompts
* `server.py`: HTTP API server (typically using FastAPI + Uvicorn or similar)
* `send.py`: Simple client script demonstrating how to send requests to `server.py`
* `finetune_fatcat.py`: Example fine-tuning script (optional, depends on environment)
* `setup.py`: Packaging / installation script (optional)

---

## Requirements

* Python 3.10+

  * Example environment includes Python 3.13
* Recommended: use a virtual environment (venv, virtualenv)

Common dependencies (depending on usage):

* `fastapi`, `uvicorn` (for API server)
* `requests` (for `send.py`)
* `torch`, `transformers`, `datasets` (for fine-tuning or PyTorch-based models)

---

## Installation (Example)

Create a virtual environment and install dependencies:

```bash
python3 -m venv env
source env/bin/activate

# Recommended: create requirements.txt or install manually
pip install fastapi uvicorn requests

# If fine-tuning or using transformer models:
pip install torch transformers datasets
```

---

## Quick Start (Local Server & Client)

### 1. Start the server (FastAPI + Uvicorn example)

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Tip: If using GPU and you need to specify device ID:

```bash
CUDA_VISIBLE_DEVICES=0 uvicorn server:app --host 0.0.0.0 --port 8000
```

---

### 2. Send a test request (in another terminal)

```bash
python send.py --host 127.0.0.1 --port 8000 --message "Tell me a fatcat joke"
```

Or using curl:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What should I eat for dinner today?"}], "temperature":0.5}'
```

---

## API Specification (Expected Format)

### POST `/chat`

#### Request JSON:

```json
{
  "messages": [
    {"role": "user", "content": "user_id: How are you today?"}
  ],
  "max_new_tokens": 128,
  "temperature": 0.3,
  "top_p": 0.95
}
```

#### Response JSON (example):

```json
{
  "status": "success",
  "response": "Generated FatCat response text",
  "metadata": { "duration_ms": 120 }
}
```

---

## Main Script Overview

### `server.py`

* Loads the model at startup (typically onto GPU)
* Provides a persistent HTTP API for repeated requests
* Note: Large models may consume significant GPU memory; ensure resources are freed when shutting down

---

### `send.py`

* Demonstrates how to call the API server
* Can be extended into a CLI tool (host/port options, timeout, output formatting, etc.)

---

### `finetune_fatcat.py`

* Used for fine-tuning the model
* Requires prepared training data (e.g., JSONL format)
```jsonl
{"messages":[{"role":"system","content":"你是 fat_cat_is_fat，一個 Discord 活躍群友。你講話直接、帶有主觀情緒，一定會給出讓群友心情更糟糕的回應。你講話一定要臭人，並且一定會具體且精確的回覆上文中的對話。請不要重複問題本身。"},{"role":"user","content":"prompt for the chatbot"},{"role":"assistant","content":"ideal response by chatbot"}]}
```
* Requires `transformers`, `datasets`, and `torch`
* Training may require significant GPU resources and time

---

### `fatcat_gpt/core.py`

* Core logic for model interaction:

  * Model initialization
  * Tokenization
  * Text generation
  * Prompt template handling
  * Phrase file integration (`fatcat_phrases.json`)
* Recommended to read inline documentation for input/output formats

---

## Configuration & Customization

* Adjust generation behavior using parameters like:

  * `temperature`
  * `top_p`
  * `max_new_tokens`
* Modify or extend prompt templates in `fatcat_phrases.json` to change model personality or response style
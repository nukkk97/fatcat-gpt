import json
import torch
import os
import random
import re

from unsloth import FastLanguageModel
from peft import PeftModel
import opencc  # pip install opencc-python-reimplemented

SYSTEM_PROMPT = {
    "role": "system",
    "content": "你是 fat_cat_is_fat，一個 Discord 活躍群友。你講話直接、帶有主觀情緒，一定會給出讓群友心情更糟糕的回應。你講話一定要臭人，並且一定會具體且精確的回覆上文中的對話。請不要重複問題本身。"
}

class FatCatGPT:
    def __init__(
        self, 
        base_model="unsloth/gpt-oss-20b", 
        lora_repo="/mnt/raid0/home/nukliliang/fatcat_gpt/output_optdata/checkpoint-891",
        max_seq_len=1024,
        load_in_4bit=True
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.converter = opencc.OpenCC("s2twp")  # 簡體 → 臺灣繁體（含詞彙替換）
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        phrase_path = os.path.join(current_dir, "fatcat_phrases.json")
        
        with open(phrase_path, "r", encoding="utf-8") as f:
            phrase_data = json.load(f)
        self.top_phrases = [p[0] for p in phrase_data["top_phrases"]]

        print("Loading Base Model...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_len,
            load_in_4bit=load_in_4bit,
            dtype=None,
        )

        print(f"Loading LoRA from {lora_repo}...")
        self.model = PeftModel.from_pretrained(self.model, lora_repo)
        
        FastLanguageModel.for_inference(self.model)
        print("FatCatGPT 初始化完成！")

    def _to_traditional(self, text: str) -> str:
        """簡體轉繁體（臺灣用詞）"""
        return self.converter.convert(text)

    def chat(self, messages, max_new_tokens=128, temperature=0.35, top_p=0.95):
        """
        傳入對話歷史 (messages)，回傳生成的字串。
        格式範例: [{"role": "user", "content": "user123: 你好"}]
        """
        # 把 user_id: 前綴拿掉，不修改原始 list
        cleaned_messages = [
            {**msg, "content": re.sub(r"^\S+:\s*", "", msg["content"])}
            if msg.get("role") == "user" else msg
            for msg in messages
        ]

        # 注入 system prompt
        full_messages = cleaned_messages

        prompt = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # prefix injection
        injected_phrase = ""
        if len(self.top_phrases) > 0 and random.random() < 0.5:
            injected_phrase = random.choice(self.top_phrases)
            prompt += injected_phrase + " "

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.model.device)

        # text generation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=1.1,
                use_cache=True,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        raw_output = (
            f"{injected_phrase} {generated_text}".replace("assistant", "").strip()
            if injected_phrase
            else generated_text.replace("assistant", "").strip()
        )

        # 簡轉繁
        final_output = self._to_traditional(raw_output)
        
        return final_output.strip()
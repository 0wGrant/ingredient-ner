import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from ingredient_ner.allergens import Linker, analyze, load_table
from ingredient_ner.context import LEXICON
from ingredient_ner.dictionary import lemmatizer
from ingredient_ner.ner import predict_text
from ingredient_ner.paths import MODELS
from ingredient_ner.recipes import load_recipes

PAGE = Path(__file__).with_name("demo.html")


def make_handler(analyze_text):
    """Обработчик: страница по GET /, разбор текста по POST /analyze"""

    class Handler(BaseHTTPRequestHandler):
        def send(self, code: int, body: bytes, content_type: str):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            else:
                self.send(404, b"", "text/plain")

        def do_POST(self):
            if self.path != "/analyze":
                self.send(404, b"", "text/plain")
                return
            text = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))["text"]
            self.send(200, json.dumps(analyze_text(text), ensure_ascii=False).encode("utf-8"),
                      "application/json; charset=utf-8")

        def log_message(self, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Страница: ингредиенты и аллергены в тексте блюда")
    parser.add_argument("--model", default="variant_B_seed42", help="папка в data/models")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODELS / args.model)
    model = AutoModelForTokenClassification.from_pretrained(MODELS / args.model).to(device)
    names = {name for recipe in load_recipes() for name in recipe["ingredients"]}
    linker = Linker(names, lemmatizer(names))
    table = load_table()

    def analyze_text(text: str) -> dict:
        return analyze(text, predict_text(model, tokenizer, text, device), linker, table, LEXICON)

    print(f"http://127.0.0.1:{args.port}", flush=True)
    HTTPServer(("127.0.0.1", args.port), make_handler(analyze_text)).serve_forever()


if __name__ == "__main__":
    main()

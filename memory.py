import os
import json

class Memory:
    """Simple persistent memory for agent conversations and results."""
    def __init__(self, path: str = "agent_memory.json"):
        self.path = path
        self.data = {"history": [], "last_result": None, "last_image_path": None}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {"history": [], "last_result": None, "last_image_path": None}

    def save(self):
        try:
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ---- History helpers ----
    def add_message(self, role: str, content: str):
        self.data.setdefault("history", []).append({"role": role, "content": content})
        self.save()

    def get_history(self):
        return self.data.get("history", [])

    # ---- Result helpers ----
    def set_last_result(self, result_text: str, image_path: str):
        self.data["last_result"] = result_text
        self.data["last_image_path"] = image_path
        self.save()

    def get_last_result(self):
        return self.data.get("last_result"), self.data.get("last_image_path")

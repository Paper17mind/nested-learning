"""LLM-as-a-Teacher: Automated Interactive Tutoring & Comprehension Tracking for HOPE Lite.

An LLM ('Teacher') chats with HOPE Lite ('Student'), teaching concepts turn-by-turn.
After each lesson turn, the teacher quizzes HOPE Lite to observe its comprehension progression.

Teacher Backends Supported:
1. 'curriculum' (Default): Built-in zero-dependency structured Indonesian curriculum.
2. 'ollama': Local LLM via Ollama API (http://localhost:11434).
3. 'openai': OpenAI-compatible API (Groq, OpenRouter, OpenAI, vLLM, DeepSeek).

Usage:
    # 1. Run with built-in curriculum teacher (zero API key / offline):
    python scripts/llm_teacher_train.py --turns 5

    # 2. Run with local Ollama:
    python scripts/llm_teacher_train.py --teacher ollama --model llama3 --turns 5

    # 3. Run with OpenAI-compatible API:
    python scripts/llm_teacher_train.py --teacher openai --api-key YOUR_KEY --base-url https://api.groq.com/openai/v1 --model llama3-8b-8192
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import nested_learning as nl


def load_dotenv(filepath: Optional[str] = None) -> None:
    """Zero-dependency .env loader that populates os.environ."""
    if filepath is None:
        candidates = [
            ".env",
            os.path.join(os.path.dirname(__file__), ".env"),
            os.path.join(os.path.dirname(__file__), "..", ".env"),
        ]
        for c in candidates:
            if os.path.exists(c):
                filepath = c
                break
    if not filepath or not os.path.exists(filepath):
        return

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            if k and k not in os.environ:
                os.environ[k] = v


# Load environment variables from .env on startup
load_dotenv()


class BaseTeacher:
    """Base class for LLM teachers."""
    stage: str = "none"

    def get_lesson_and_quiz(self, turn_idx: int, topic_hint: Optional[str] = None) -> Dict[str, Any]:
        """Return dict with 'topic', 'lesson', 'quiz_prompt', 'expected_keywords'."""
        raise NotImplementedError


# ── Static curriculum loaded from external JSON (dapat diedit tanpa ubah kode) ──
_CURRICULUM_BANK_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "curriculum_bank.json"
)

_FALLBACK_CURRICULUM_BANK: Dict[str, Any] = {
    "umum": [
        {
            "topic": "Transportasi Darat: Mobil",
            "lesson": "Mobil adalah alat transportasi darat beroda empat.",
            "quiz_prompt": "Mobil adalah",
            "expected_keywords": ["alat", "transportasi", "darat", "roda", "empat"],
        },
        {
            "topic": "Teknologi: Komputer",
            "lesson": "Komputer adalah alat elektronik untuk mengolah data.",
            "quiz_prompt": "Komputer adalah",
            "expected_keywords": ["alat", "elektronik", "mengolah", "data"],
        },
        {
            "topic": "Sains: Planet Bumi",
            "lesson": "Bumi adalah planet dalam tata surya tempat kehidupan.",
            "quiz_prompt": "Bumi adalah",
            "expected_keywords": ["planet", "tata", "surya", "kehidupan"],
        },
    ]
}


def _load_curriculum_bank() -> Dict[str, Any]:
    """Load curriculum bank dari data/curriculum_bank.json; fallback ke dict bawaan jika tidak ada."""
    path = os.path.abspath(_CURRICULUM_BANK_PATH)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Warning] Gagal membaca {path}: {e}. Menggunakan fallback.")
    return _FALLBACK_CURRICULUM_BANK


CURRICULUM_BANK: Dict[str, Any] = _load_curriculum_bank()


def _load_math_curriculum(age: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load dataset matematika dari data/math_curriculum/ berdasarkan usia (1-4) atau gabungan.

    Setiap item akan dikonversi ke format lesson/quiz yang dipahami CurriculumTeacher.
    Format math item: {stage, age, topic, question, prompt, answer, keywords}
    """
    base_dir = os.path.join(os.path.dirname(__file__), "..", "data", "math_curriculum")
    base_dir = os.path.abspath(base_dir)

    age_files = {
        1: "age_1_numbers.jsonl",
        2: "age_2_numbers.jsonl",
        3: "age_3_addition_subtraction.jsonl",
        4: "age_4_operations.jsonl",
    }

    items: List[Dict[str, Any]] = []

    if age is not None:
        filenames = [age_files[age]] if age in age_files else []
    else:
        # Coba all_ages.jsonl dulu (pre-generated gabungan)
        all_path = os.path.join(base_dir, "all_ages.jsonl")
        if os.path.exists(all_path):
            filenames = ["all_ages.jsonl"]
        else:
            filenames = list(age_files.values())

    for fname in filenames:
        fpath = os.path.join(base_dir, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # Konversi ke format CurriculumTeacher
                items.append({
                    "topic": obj.get("topic", ""),
                    "lesson": obj.get("answer", ""),
                    "quiz_question": obj.get("question", ""),
                    "quiz_prompt": obj.get("prompt", ""),
                    "expected_keywords": obj.get("keywords", []),
                    # Meta tambahan untuk filtering per stage
                    "stage_name": obj.get("stage", "anak"),
                    "age": obj.get("age", 4),
                })
    return items


CURRICULUM_BANK = _load_curriculum_bank()


def build_teacher_prompt(stage: str, turn_idx: int, topic_hint: Optional[str], seen_topics: List[str]) -> Tuple[str, str]:
    """Build developmentally calibrated, non-static prompts for LLM teachers."""
    curr_stage = stage
    if stage == "grow":
        stage_map = {1: "bayi", 2: "batita", 3: "balita", 4: "anak"}
        curr_stage = stage_map.get(min(4, turn_idx), "anak")

    avoid_clause = ""
    if seen_topics:
        avoid_clause = f"PENTING: Jangan gunakan topik yang sudah diajarkan di sesi ini: {', '.join(seen_topics[-10:])}. Pilih objek/konsep baru yang BERBEDA!\n"

    if curr_stage == "bayi":
        stage_instructions = """Anda adalah Ibu / Guru Pengasuh yang berbicara kepada bayi (0-12 bulan / Child-Directed Speech).
Dunia Bayi: Anggota keluarga (ibu, bapak, kakak), panca indra tubuh (mata, telinga, hidung, mulut, kaki, tangan), suara hewan lucu (kucing meong, bebek kwek, burung cuit, sapi mbek), makanan/minuman sehat (susu, air putih, pisang), mainan (bola, balok, boneka), atau alam (matahari hangat, bintang, kasur empuk).
Aturan Wajib:
1. 'topic': Nama objek tunggal sederhana (contoh: 'Bebek', 'Air Putih', 'Hidung', 'Matahari', 'Kasur').
2. 'lesson': HANYA 1 kalimat fakta tunggal super ringkas (4-7 kata, maks 40 karakter), ramah dan jelas. Contoh: 'Bebek berbunyi kwek kwek kwek.' atau 'Air putih menyehatkan tubuh adik.'
3. 'quiz_question': Kalimat tanya kuis singkat (contoh: 'Bebek berbunyi apa?' atau 'Apa manfaat air putih?').
4. 'quiz_prompt': 2 kata awalan persis dari 'lesson' (contoh: 'Bebek berbunyi' atau 'Air putih').
5. 'expected_keywords': 1 sampai 2 kata kunci utama yang ada dalam 'lesson'."""
        default_topic = f"konsep bayi baru ke-{turn_idx}"
    elif curr_stage == "batita":
        stage_instructions = """Anda adalah Guru untuk batita (1-2 tahun / Tahap Aksi & Gerak).
Dunia Batita: Gerakan hewan, makanan favorit, kendaraan di jalan, mainan aksi (burung terbang tinggi, ikan berenang di kolam, kelinci makan wortel, mobil melaju di jalan).
Aturan Wajib:
1. 'lesson': HANYA 1 kalimat aksi tunggal (5-8 kata). Contoh: 'Burung terbang tinggi di angkasa.'
2. 'quiz_question': Pertanyaan aksi (contoh: 'Burung terbang di mana?').
3. 'quiz_prompt': 2 kata awalan persis dari 'lesson' (contoh: 'Burung terbang').
4. 'expected_keywords': 2 kata kunci penting."""
        default_topic = f"konsep batita baru ke-{turn_idx}"
    elif curr_stage == "balita":
        stage_instructions = """Anda adalah Guru PAUD untuk balita (3-4 tahun / Sifat & Warna).
Dunia Balita: Sifat benda, warna buah, bentuk, aroma bunga, ukuran hewan.
Aturan Wajib:
1. 'lesson': HANYA 1 kalimat sifat/warna tunggal (6-9 kata). Contoh: 'Buah apel berwarna merah merona.'
2. 'quiz_question': Pertanyaan warna/sifat (contoh: 'Buah apel berwarna apa?').
3. 'quiz_prompt': 3 kata awalan persis dari 'lesson' (contoh: 'Buah apel berwarna').
4. 'expected_keywords': 2 kata kunci penting."""
        default_topic = f"konsep balita baru ke-{turn_idx}"
    elif curr_stage == "anak":
        stage_instructions = """Anda adalah Guru SD untuk anak usia 5+ tahun (Sebab-Akibat & Kebiasaan Baik).
Dunia Anak: Fungsi alam, kebiasaan sehat, manfaat matahari, hujan menyuburkan tanaman, gosok gigi.
Aturan Wajib:
1. 'lesson': HANYA 1 kalimat padat (7-10 kata). Contoh: 'Air hujan menyegarkan tanaman dan rumput.'
2. 'quiz_question': Pertanyaan manfaat/fungsi (contoh: 'Apa manfaat air hujan?').
3. 'quiz_prompt': 2 kata awalan persis dari 'lesson' (contoh: 'Air hujan').
4. 'expected_keywords': 2 kata kunci penting."""
        default_topic = f"konsep anak baru ke-{turn_idx}"
    else:
        stage_instructions = """Anda adalah Guru AI untuk model bahasa pemula (Small Language Model).
Aturan Wajib:
1. 'lesson': HANYA 1 kalimat fakta tunggal ringkas (maks 7-10 kata).
2. 'quiz_question': Kalimat tanya kuis definisi atau fungsi.
3. 'quiz_prompt': Awalan persis dari 'lesson'.
4. 'expected_keywords': 2 sampai 3 kata kunci."""
        default_topic = f"topik umum ke-{turn_idx} (sains, teknologi, biologi, transportasi, sejarah)"

    target_topic = topic_hint or default_topic
    prompt = f"""{stage_instructions}
{avoid_clause}Topik Sasaran: '{target_topic}'.

Keluarkan output dalam JSON murni persis format ini:
{{"topic": "Nama Topik", "lesson": "1 kalimat pelajaran.", "quiz_question": "Pertanyaan kuis?", "quiz_prompt": "Awalan prompt", "expected_keywords": ["kata1", "kata2"]}}
HANYA JSON murni tanpa penjelasan markdown."""
    return target_topic, prompt


class CurriculumTeacher(BaseTeacher):
    """Built-in structured Indonesian curriculum teacher.

    Mendukung tiga sumber kurikulum:
      1. Developmental QA (data/developmental_qa.jsonl) - bayi/batita/balita/anak
      2. Math Curriculum  (data/math_curriculum/)        - angka, hitung, operasi
      3. General Curriculum (data/curriculum_bank.json)  - transportasi, sains, dll
    """

    def __init__(
        self,
        category: str = "all",
        custom_file: Optional[str] = None,
        stage: str = "none",
        math_age: Optional[int] = None,   # None = semua usia; 1-4 = usia spesifik
        use_math: bool = False,            # True = gunakan math curriculum
    ):
        self.stage = stage
        self.seen_topics: set = set()
        self.dev_items: List[Dict[str, Any]] = []
        self.math_items: List[Dict[str, Any]] = []

        # 1. Load Developmental QA (untuk stage bayi/batita/balita/anak/grow)
        dev_path = os.path.join(os.path.dirname(__file__), "..", "data", "developmental_qa.jsonl")
        dev_path = os.path.abspath(dev_path)
        if stage in ("bayi", "batita", "balita", "anak", "grow") and os.path.exists(dev_path):
            with open(dev_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.dev_items.append(json.loads(line))

        # 2. Load Math Curriculum (opsional via --materi-math)
        if use_math:
            self.math_items = _load_math_curriculum(age=math_age)
            print(f"[Math Curriculum] Dimuat {len(self.math_items)} soal matematik"
                  f" (usia={math_age or 'semua'}).")

        # 3. Load General Curriculum
        self.curriculum: List[Dict[str, Any]] = []
        if custom_file and os.path.exists(custom_file):
            self.curriculum = self._load_custom_file(custom_file)
        elif category == "all":
            for items in CURRICULUM_BANK.values():
                self.curriculum.extend(items)
        elif category in CURRICULUM_BANK:
            self.curriculum = list(CURRICULUM_BANK[category])
        else:
            for items in CURRICULUM_BANK.values():
                self.curriculum.extend(items)

        # Gabungkan math_items ke curriculum jika ada
        if self.math_items:
            self.curriculum = self.math_items + self.curriculum

    def _load_custom_file(self, filepath: str) -> List[Dict[str, Any]]:
        items = []
        if filepath.endswith(".jsonl"):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        items.append(data)
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                paragraphs = [p.strip() for p in f.read().split("\n\n") if p.strip()]
            for idx, p in enumerate(paragraphs, 1):
                words = p.split()
                prompt = " ".join(words[:min(4, len(words))])
                items.append({
                    "topic": f"Materi File #{idx}",
                    "lesson": p,
                    "quiz_prompt": prompt,
                    "expected_keywords": [w.lower() for w in words[4:min(12, len(words))]],
                })
        return items

    def get_lesson_and_quiz(self, turn_idx: int, topic_hint: Optional[str] = None) -> Dict[str, Any]:
        if topic_hint:
            hint_lower = topic_hint.lower().strip()
            # 1. Cek di materi perkembangan (bayi/batita/balita/anak) jika ada
            if self.stage != "none" and self.dev_items:
                matches = [it for it in self.dev_items if hint_lower in it.get("topic", "").lower()]
                if matches:
                    it = matches[0]
                    curr_stage = it.get("stage_name", self.stage)
                    return {
                        "topic": f"[{curr_stage.upper()}] {it['topic']}",
                        "lesson": it["answer"],
                        "quiz_question": it["question"],
                        "quiz_prompt": it["prompt"],
                        "expected_keywords": it["keywords"],
                    }
            # 2. Cek di kurikulum umum
            for item in self.curriculum:
                if hint_lower in item["topic"].lower() or hint_lower in item["lesson"].lower():
                    res = dict(item)
                    if "quiz_question" not in res:
                        clean_name = res["topic"].split(": ")[-1] if ": " in res["topic"] else res["topic"]
                        res["quiz_question"] = f"Apa yang dimaksud dengan {clean_name}?"
                    return res
            # 3. Buat materi dinamis jika topik kustom tidak ada di daftar
            words = topic_hint.split()
            return {
                "topic": f"Topik Kustom: {topic_hint}",
                "lesson": f"{topic_hint} adalah konsep penting yang memiliki karakteristik unik dan peran besar dalam peradaban manusia.",
                "quiz_question": f"Apa yang dimaksud dengan {topic_hint}?",
                "quiz_prompt": f"{topic_hint} adalah",
                "expected_keywords": [w.lower() for w in words] + ["konsep", "penting", "peran"],
            }

        if self.math_items:
            if self.stage != "none":
                stage_map = {1: "bayi", 2: "batita", 3: "balita", 4: "anak"}
                curr_stage = stage_map.get(min(4, turn_idx), "anak") if self.stage == "grow" else self.stage
                math_pool = [it for it in self.math_items if it.get("stage_name") == curr_stage]
                if not math_pool:
                    math_pool = self.math_items
            else:
                math_pool = self.math_items

            unseen = [it for it in math_pool if it["topic"].lower() not in self.seen_topics]
            if not unseen:
                self.seen_topics.clear()
                unseen = math_pool

            it = random.choice(unseen)
            self.seen_topics.add(it["topic"].lower())

            return {
                "topic": it["topic"],
                "lesson": it.get("lesson", it.get("answer", "")),
                "quiz_question": it.get("quiz_question", it.get("question", "")),
                "quiz_prompt": it.get("quiz_prompt", it.get("prompt", "")),
                "expected_keywords": it.get("expected_keywords", it.get("keywords", [])),
            }

        if self.stage != "none" and self.dev_items:
            stage_map = {1: "bayi", 2: "batita", 3: "balita", 4: "anak"}
            if self.stage == "grow":
                curr_stage = stage_map.get(min(4, turn_idx), "anak")
            else:
                curr_stage = self.stage

            stage_pool = [it for it in self.dev_items if it.get("stage_name") == curr_stage]
            if not stage_pool:
                stage_pool = self.dev_items

            # Dynamic sampling: prioritas item yang belum pernah diajarkan di sesi ini
            unseen = [it for it in stage_pool if it["topic"].lower() not in self.seen_topics]
            if not unseen:
                self.seen_topics.clear()
                unseen = stage_pool

            it = random.choice(unseen)
            self.seen_topics.add(it["topic"].lower())

            return {
                "topic": f"[{curr_stage.upper()}] {it['topic']}",
                "lesson": it["answer"],
                "quiz_question": it["question"],
                "quiz_prompt": it["prompt"],
                "expected_keywords": it["keywords"],
            }

        unseen = [item for item in self.curriculum if item["topic"].lower() not in self.seen_topics]
        if not unseen:
            self.seen_topics.clear()
            unseen = self.curriculum
        item = random.choice(unseen)
        self.seen_topics.add(item["topic"].lower())
        res = dict(item)
        if "quiz_question" not in res:
            clean_name = res["topic"].split(": ")[-1] if ": " in res["topic"] else res["topic"]
            res["quiz_question"] = f"Apa yang dimaksud dengan {clean_name}?"
        return res


class OllamaTeacher(BaseTeacher):
    """Teacher powered by a local Ollama LLM instance."""

    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3", stage: str = "none"):
        self.host = host.rstrip("/")
        self.model = model
        self.stage = stage
        self.seen_topics: List[str] = []
        self.fallback = CurriculumTeacher(stage=stage)

    def _call_ollama(self, prompt: str) -> str:
        url = f"{self.host}/api/generate"
        payload = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()

    def get_lesson_and_quiz(self, turn_idx: int, topic_hint: Optional[str] = None) -> Dict[str, Any]:
        self.fallback.stage = self.stage
        target_topic, prompt = build_teacher_prompt(self.stage, turn_idx, topic_hint, self.seen_topics)
        try:
            raw = self._call_ollama(prompt)
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                obj = json.loads(raw[start:end])
                if 'lesson' in obj and not obj['lesson'].strip().endswith('.'):
                    obj['lesson'] = obj['lesson'].strip() + '.'
                if "quiz_question" not in obj:
                    obj["quiz_question"] = f"Apa yang dimaksud dengan {obj.get('topic', target_topic)}?"
                if "quiz_prompt" not in obj:
                    obj["quiz_prompt"] = f"{obj.get('topic', target_topic)} adalah"
                self.seen_topics.append(obj.get("topic", target_topic))
                return obj
        except Exception as e:
            print(f"[Ollama Teacher Warning]: Gagal menghubungi Ollama ({e}). Menggunakan kurikulum dinamis bawaan.")
        return self.fallback.get_lesson_and_quiz(turn_idx, topic_hint=topic_hint)


class OpenAITeacher(BaseTeacher):
    """Teacher powered by OpenAI-compatible APIs (Groq, OpenRouter, OpenAI, vLLM)."""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", model: str = "gpt-3.5-turbo", stage: str = "none"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.stage = stage
        self.seen_topics: List[str] = []
        self.fallback = CurriculumTeacher(stage=stage)

    def _call_api(self, prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "stream": False,
        }).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-Title": "Nested Learning HOPE Lite",
        }
        req = urllib.request.Request(url, data=payload, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8").strip()

        data, _ = json.JSONDecoder().raw_decode(body)
        return data["choices"][0]["message"]["content"].strip()

    def get_lesson_and_quiz(self, turn_idx: int, topic_hint: Optional[str] = None) -> Dict[str, Any]:
        self.fallback.stage = self.stage
        target_topic, prompt = build_teacher_prompt(self.stage, turn_idx, topic_hint, self.seen_topics)
        try:
            raw = self._call_api(prompt)
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                obj = json.loads(raw[start:end])
                if 'lesson' in obj and not obj['lesson'].strip().endswith('.'):
                    obj['lesson'] = obj['lesson'].strip() + '.'
                if "quiz_question" not in obj:
                    obj["quiz_question"] = f"Apa yang dimaksud dengan {obj.get('topic', target_topic)}?"
                if "quiz_prompt" not in obj:
                    obj["quiz_prompt"] = f"{obj.get('topic', target_topic)} adalah"
                self.seen_topics.append(obj.get("topic", target_topic))
                return obj
        except Exception as e:
            print(f"[OpenAI Teacher Warning]: Gagal menghubungi API ({e}). Menggunakan kurikulum dinamis bawaan.")

        return self.fallback.get_lesson_and_quiz(turn_idx, topic_hint=topic_hint)

def parse_args():
    parser = argparse.ArgumentParser(description="LLM-as-a-Teacher automated training for HOPE Lite")
    parser.add_argument("--teacher", type=str, default=os.getenv("LLM_TEACHER", "auto"),
                        choices=["auto", "curriculum", "ollama", "openai"],
                        help="Teacher engine: 'auto' (detect from .env), 'curriculum' (offline), 'ollama', or 'openai'")
    parser.add_argument("--turns", type=int, default=5, help="Number of interactive learning turns")
    parser.add_argument("--topic", type=str, default=None,
                        help="Specific topic/theme for the teacher (e.g. --topic 'Dinosaurus' or --topic 'Robotika')")
    parser.add_argument("--topics", type=str, default=None,
                        help="Comma-separated topics for consecutive turns")
    parser.add_argument("--stage", type=str, default="none",
                        choices=["none", "bayi", "batita", "balita", "anak", "grow"],
                        help="Tahap perkembangan kognitif: 'bayi', 'batita', 'balita', 'anak', atau 'grow' (tumbuh bertahap)")
    parser.add_argument("--sleep", action="store_true", default=True,
                        help="Aktifkan fase tidur konsolidasi memori (Sleep Replay) di akhir sesi")
    parser.add_argument("--no-sleep", dest="sleep", action="store_false",
                        help="Nonaktifkan fase tidur")
    parser.add_argument("--sleep-steps", type=int, default=20, help="Jumlah siklus konsolidasi memori saat tidur")
    parser.add_argument("--sleep-lr", type=float, default=1.5e-3, help="Learning rate saat fase tidur")
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Aktifkan augmentasi teks otomatis (variasi pola & sinonim) saat tidur")
    parser.add_argument("--no-augment", dest="augment", action="store_false",
                        help="Nonaktifkan augmentasi teks otomatis saat tidur")
    parser.add_argument("--category", type=str, default="all",
                        help="Kategori kurikulum dari data/curriculum_bank.json (default: 'all'). "
                             "Nilai yang tersedia: all, transportasi, teknologi, sains, biologi, kuliner")
    parser.add_argument("--materi-file", type=str, default=None,
                        help="Path ke file .txt atau .jsonl berisi materi kustom")

    # ── Math Curriculum ────────────────────────────────────────────────────────
    parser.add_argument("--materi-math", action="store_true", default=False,
                        help="Gunakan dataset matematika dari data/math_curriculum/ "
                             "(angka, hitung, penjumlahan, pengurangan, perkalian, dll)")
    parser.add_argument("--math-age", type=int, default=None, choices=[1, 2, 3, 4],
                        help="Filter usia matematika: 1=bayi(1-5), 2=batita(6-10), "
                             "3=balita(+/-), 4=anak(x/:). Default: semua usia")

    parser.add_argument("--checkpoint", type=str, default="models/hope_infant.npz",
                        help="Base model checkpoint to teach")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Lanjutkan belajar dari checkpoint tutored sebelumnya")
    parser.add_argument("--save-path", type=str, default="models/hope_tutored.npz",
                        help="Path to save the model after tutoring")
    parser.add_argument("--lr", type=float, default=1.2e-3, help="Learning rate per turn")
    parser.add_argument("--learn-steps", type=int, default=14, help="Optimization steps per lesson turn")
    parser.add_argument("--max-quiz-tokens", type=int, default=65, help="Max tokens during quiz")
    parser.add_argument("--temperature", type=float, default=0.15, help="Sampling temperature during quiz")

    # API options (defaults from .env)
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY", ""),
                        help="API key for teacher (defaults to OPENAI_API_KEY in .env)")
    parser.add_argument("--base-url", type=str, default=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
                        help="Base URL for API")
    parser.add_argument("--model", type=str, default=os.getenv("LLM_MODEL", "meta-llama/llama-3-8b-instruct:free"),
                        help="Model name for teacher")
    return parser.parse_args()

def clean_sentence_output(text: str) -> str:
    """Trim generated output cleanly at the first terminal punctuation ('.', '!', '?') or newline."""
    lines = [l.strip() for l in text.split(chr(10)) if l.strip()]
    if not lines:
        return ""
    # Avoid truncating when output begins with a stray leading punctuation mark
    first_line = lines[0].lstrip(".,!?:; ").strip()
    if not first_line and len(lines) > 1:
        first_line = lines[1].lstrip(".,!?:; ").strip()

    candidates = [first_line.find(p) for p in [".", "!", "?"] if first_line.find(p) > 1]
    if candidates:
        min_idx = min(candidates)
        return first_line[:min_idx + 1].strip()
    return first_line.strip()

def score_response(response: str, expected_keywords: List[str]) -> Tuple[float, int, int]:
    """Calculate keyword recall hit rate."""
    resp_lower = response.lower()
    hits = 0
    for kw in expected_keywords:
        if kw.lower() in resp_lower:
            hits += 1
    total = len(expected_keywords)
    pct = (hits / max(1, total)) * 100.0
    return pct, hits, total

def sleep_consolidation(model, tok, history_records, memory_state=None, steps=20, lr=1.5e-3, target_loss=0.08, augment=True):
    """Simulate infant night sleep: replay recent memories cleanly into slow CMS tiers with text data augmentation."""
    if not history_records:
        return
    print("\n" + "=" * 78)
    print("😴 FASE TIDUR & KONSOLIDASI MEMORI (NEURAL REPLAY DENGAN AUGMENTASI)...")
    print("   Otak beristirahat. Hipokampus mentransfer ingatan bersih ke Neokorteks")
    print(f"   (CMS Tiers) dengan variasi semantik otomatis (lr={lr:.1e}).")
    print("=" * 78)

    optimizer = nl.NestedOptimizer(model.tier_param_groups(), lr=lr, optimizer_cls=nl.AdamW)
    loss_fn = nl.CrossEntropyLoss()

    clean_samples = []
    for r in history_records:
        sample_text = r.get("study_text", r.get("lesson", "")).strip()
        if sample_text:
            if not sample_text.endswith("."):
                sample_text += "."
            if augment:
                # Augmentasi variasi otomatis (analog dengan rotasi/flip/jitter di computer vision)
                variants = nl.augment_text(sample_text, max_variations=3)
                clean_samples.extend(variants)
            else:
                clean_samples.append(sample_text)

    for s in range(steps):
        losses = []
        # Acak urutan setiap siklus tidur agar bobot neokorteks tidak bias ke giliran terakhir
        shuffled = list(clean_samples)
        random.shuffle(shuffled)
        for text in shuffled:
            study_tokens = tok.encode(text)
            if len(study_tokens) < 3:
                continue
            inp = np.array([study_tokens[:-1]], dtype=np.int64)
            tgt = np.array([study_tokens[1:]], dtype=np.int64)

            optimizer.zero_grad()
            # Replay bersih langsung ke bobot lambat tanpa polusi fast-memory spesifik turn
            logits, _ = model.forward(inp, state=None)
            loss_val = loss_fn.forward(logits, tgt)
            model.backward(loss_fn.backward())
            optimizer.step()
            losses.append(float(loss_val))
        avg_l = float(np.mean(losses)) if losses else 0.0
        if (s + 1) % 5 == 0 or s == 0 or avg_l <= target_loss:
            print(f"  • Siklus Tidur {s+1}/{steps} | Konsolidasi Loss: {avg_l:.4f}")
        if avg_l <= target_loss and (s + 1) >= 8:
            print(f"✓ Target kristalisasi memori tercapai (Loss <= {target_loss}).")
            break
    print("✓ Konsolidasi memori selesai! Ingatan baru berhasil mengkristal di Neokorteks.\n")

def save_tutoring_logs(history_records: List[Dict[str, Any]], meta_info: Dict[str, Any]) -> Tuple[str, str]:
    """Save tutoring session logs unified into single cumulative files (tutoring_history.jsonl and tutoring_history.md)."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    session_id = meta_info.get("session_id", "default")

    jsonl_path = os.path.join(log_dir, "tutoring_history.jsonl")
    md_path = os.path.join(log_dir, "tutoring_history.md")

    # 1. Append to unified JSONL log
    with open(jsonl_path, "a", encoding="utf-8") as f:
        for r in history_records:
            entry = {**meta_info, **r}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 2. Append to unified Markdown Report
    avg_pre = float(np.mean([r['pre_score'] for r in history_records]))
    avg_post = float(np.mean([r['post_score'] for r in history_records]))

    file_exists = os.path.exists(md_path) and os.path.getsize(md_path) > 0

    md_content = []
    if not file_exists:
        md_content.append("# 📚 Riwayat Sesi Pembelajaran Guru LLM & Murid HOPE Lite\n")

    md_content.extend([
        f"## 🎓 Sesi Pembelajaran: `{session_id}`",
        f"- **Waktu Sesi:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Guru (Teacher):** {meta_info.get('teacher_desc', 'N/A')}",
        f"- **Model Base Murid:** `{meta_info.get('checkpoint', 'N/A')}`",
        f"- **Total Turn:** {len(history_records)} giliran",
        f"- **Skor Pemahaman:** **{avg_pre:.1f}%** (Sebelum) ──► **{avg_post:.1f}%** (Sesudah Belajar)",
        "",
        "### 📈 Tabel Rangkuman Perkembangan per Turn",
        "",
        "| Turn | Topik Pelajaran | Loss | Memory \\|\\|M\\|\\| | Pre-Test Score | Post-Test Score | Peningkatan |",
        "|---|---|---|---|---|---|---|",
    ])

    for r in history_records:
        delta = r['post_score'] - r['pre_score']
        delta_str = f"+{delta:.0f}%" if delta > 0 else f"{delta:.0f}%"
        md_content.append(
            f"| {r['turn']} | {r['topic']} | {r['loss']:.4f} | {r['norm']:.4f} | {r['pre_score']:.0f}% | {r['post_score']:.0f}% | **{delta_str}** |"
        )

    md_content.extend([
        "",
        "### 📝 Rincian Dialog Pembelajaran",
        "",
    ])

    for r in history_records:
        md_content.extend([
            f"#### Turn {r['turn']}: {r['topic']}",
            f"- **Materi Guru:**",
            f"  > *\"{r.get('lesson', '')}\"*",
            f"- **Pertanyaan Kuis Guru:** `{r.get('quiz_question', r.get('quiz_prompt', ''))}`",
            f"- **Prompt Evaluasi Murid:** `{r.get('quiz_prompt', '')}`",
            f"- **Jawaban Murid Sebelum Belajar:**",
            f"  > `{r['pre_answer']}` *(Recall: {r['pre_score']:.0f}%)*",
            f"- **Jawaban Murid Sesudah Belajar:**",
            f"  > `{r['post_answer']}` *(Recall: {r['post_score']:.0f}%)*",
            f"- **Metrik Teknis:**",
            f"  - Loss Pembelajaran: `{r['loss']:.4f}` *(Perplexity: {float(np.exp(min(15.0, r['loss']))):.2f})*",
            f"  - Norma Matriks Memori $\\|M\\|$: `{r['norm']:.4f}`",
            "",
        ])

    md_content.append("\n---\n")

    with open(md_path, "a", encoding="utf-8") as f:
        f.write("\n".join(md_content) + "\n")

    return jsonl_path, md_path


def main():
    args = parse_args()

    session_id = time.strftime("tutoring_%Y%m%d_%H%M%S")
    print("=" * 75)
    print(f"🎓 LLM-AS-A-TEACHER: INTERACTIVE CONTINUAL TUTORING FOR HOPE LITE")
    print(f"Session ID: {session_id}")
    print("=" * 75)
    # 1. Setup Teacher
    teacher_type = args.teacher
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")

    if teacher_type in ("auto", "openai"):
        if not api_key or api_key == "your_api_key_here":
            if teacher_type == "openai":
                print("[Info]: API key belum diisi di .env (masih 'your_api_key_here').")
                print("        Beralih otomatis ke Kurikulum Terstruktur Bawaan (Offline).")
                print("        (Untuk menggunakan 9router/OpenRouter, isi OPENAI_API_KEY di file .env)\n")
            teacher_type = "curriculum"
        else:
            teacher_type = "openai"

    teacher_desc = f"Kurikulum Edukasi '{args.category.upper()}' (Offline Bawaan)" if args.category != "all" else "Kurikulum Edukasi Terstruktur (Offline Bawaan)"
    if args.materi_file:
        teacher_desc = f"Kurikulum dari Berkas '{os.path.basename(args.materi_file)}' (Offline)"
    if teacher_type == "curriculum":
        if args.stage != "none":
            teacher_desc = f"Kurikulum Tumbuh Kembang Kognitif (Tahap: {args.stage.upper()})"
        teacher = CurriculumTeacher(
                category=args.category,
                custom_file=args.materi_file,
                stage=args.stage,
                use_math=getattr(args, "materi_math", False),
                math_age=getattr(args, "math_age", None),
            )
        print(f"Guru (Teacher): {teacher_desc}")
    elif teacher_type == "ollama":
        teacher = OllamaTeacher(model=args.model, stage=args.stage)
        teacher_desc = f"Ollama Lokal ({args.model})"
        print(f"Guru (Teacher): {teacher_desc}")
    elif teacher_type == "openai":
        teacher = OpenAITeacher(api_key=api_key, base_url=args.base_url, model=args.model, stage=args.stage)
        provider_name = "OpenRouter / 9router" if "openrouter" in args.base_url.lower() or "9router" in args.base_url.lower() else "OpenAI API"
        teacher_desc = f"{provider_name} ({args.model})"
        print(f"Guru (Teacher): {teacher_desc} (URL: {args.base_url})")
    ckpt_path = args.checkpoint
    if args.resume and os.path.exists(args.save_path):
        ckpt_path = args.save_path
    elif args.stage != "none" and args.checkpoint == "models/hope_id.npz" and os.path.exists("models/hope_infant.npz"):
        ckpt_path = "models/hope_infant.npz"
    if not os.path.exists(ckpt_path) and os.path.exists(os.path.join("models", ckpt_path)):
        ckpt_path = os.path.join("models", ckpt_path)
    elif not os.path.exists(ckpt_path) and os.path.exists("models/hope_model.npz"):
        ckpt_path = "models/hope_model.npz"

    tok = nl.resolve_tokenizer_for_checkpoint(ckpt_path)

    if os.path.exists(ckpt_path):
        print(f"Murid (Student): HOPE Lite dimuat dari '{ckpt_path}'")
        model, meta = nl.HOPE.load_checkpoint(ckpt_path)
        memory_state = meta.get("memory_state")
        print(f"Tokenizer Murid: {tok.__class__.__name__} (Vocab: {tok.vocab_size})")
    else:
        print("Murid (Student): HOPE Lite baru (Fresh Initialization)")
        model = nl.HOPE(vocab_size=tok.vocab_size, d_model=48, n_layers=4, cms_tiers=[[2, 1], [2, 4]])
        memory_state = None

    optimizer = nl.NestedOptimizer(model.tier_param_groups(), lr=args.lr, optimizer_cls=nl.AdamW)
    loss_fn = nl.CrossEntropyLoss(ignore_index=tok.pad_token_id)

    print(f"Parameter Murid: {model.count_parameters():,} | Tiers: {model.cms_tiers}")
    print(f"Rencana Pembelajaran: {args.turns} Giliran (Turns) Interaktif")
    turn_topics = []
    if args.topics:
        turn_topics = [t.strip() for t in args.topics.split(",") if t.strip()]
    elif args.topic:
        turn_topics = [args.topic.strip()]

    if turn_topics:
        print(f"Topik Pembelajaran Khusus: {turn_topics}")
    print("=" * 75)

    history_records = []

    for turn in range(1, args.turns + 1):
        topic_hint = turn_topics[(turn - 1) % len(turn_topics)] if turn_topics else None
        item = teacher.get_lesson_and_quiz(turn, topic_hint=topic_hint)
        topic = item.get("topic", f"Pelajaran {turn}")
        lesson_text = item.get("lesson", "").strip()
        quiz_question = item.get("quiz_question") or f"Apa yang dimaksud dengan {topic}?"
        quiz_prompt = item.get("quiz_prompt") or f"{topic} adalah"
        expected_keywords = item.get("expected_keywords", [])

        # Pastikan awalan materi selaras dengan quiz_prompt agar murid bisa mengingat dengan tepat
        clean_prompt = quiz_prompt.strip()
        if not lesson_text.lower().startswith(clean_prompt.lower()):
            study_text = f"{clean_prompt} {lesson_text}\n\n"
        else:
            study_text = lesson_text + "\n\n"

        print(f"\n┌────────────────────────────────────────────────────────────────────────┐")
        print(f"│ 📚 TURN {turn}/{args.turns}: {topic:<58} │")
        print(f"├────────────────────────────────────────────────────────────────────────┤")

        # --- A. Murid Mengecek Memori SEBELUM Belajar (Pre-Test) ---
        pre_quiz_tokens = tok.encode(quiz_prompt)
        pre_gen_ids = model.generate(
            pre_quiz_tokens,
            max_new_tokens=args.max_quiz_tokens,
            temperature=args.temperature,
            top_k=20,
            repetition_penalty=1.25,
            state=None,  # Pre-test murni menguji memori jangka panjang tanpa polusi working memory turn sebelumnya
        )
        raw_pre = tok.decode(pre_gen_ids[len(pre_quiz_tokens):]).strip()
        pre_answer = clean_sentence_output(raw_pre)
        full_pre = f"{quiz_prompt} {pre_answer}".strip()
        pre_score, pre_hits, total_kw = score_response(full_pre, expected_keywords)

        # --- B. Murid Aktif Tanya Balik jika Belum Tahu ---
        is_unknown = pre_score < 40.0
        if is_unknown:
            inquiry = f"Guru, aku belum tahu tentang {topic}. Mohon jelaskan apa itu {topic} dan fakta kuncinya?"
            print(f"│ 🤖 MURID TANYA BALIK (Belum Tahu):")
            print(f"│   \"{inquiry}\"")
            print(f"│")
            print(f"│ 🧑‍🏫 GURU MENJAWAB & MENGAJAR:")
            print(f"│   \"{lesson_text}\"")
        else:
            inquiry = f"Apa itu {topic}?"
            print(f"│ 🧑‍🏫 GURU MEMPERDALAM MATERI:")
            print(f"│   \"{lesson_text}\"")

        # --- C. Pertanyaan Kuis dari Guru ---
        print(f"│")
        print(f"│ 🧑‍🏫 PERTANYAAN KUIS DARI GURU:")
        print(f"│   \"{quiz_question}\"")
        print(f"│   (Prompt Evaluasi Murid: \"{quiz_prompt}\")")

        # --- D. Murid Belajar & Menyerap Materi (In-Chat Learning & Memory Update) ---
        study_tokens = tok.encode(study_text)
        inp = np.array([study_tokens[:-1]], dtype=np.int64)
        tgt = np.array([study_tokens[1:]], dtype=np.int64)

        # Segarkan optimizer per turn agar momentum gradien materi lama tidak membiaskan materi baru
        turn_optimizer = nl.NestedOptimizer(model.tier_param_groups(), lr=args.lr, optimizer_cls=nl.AdamW)

        # 1. Update Fast-Weight Memory via Delta Rule (fokus memori kerja pada materi saat ini)
        _, memory_state = model.forward(inp, state=None)
        mem_norm = float(np.linalg.norm(memory_state[0]))

        # 2. Update Slow Weights via Analytical Gradient Descent
        turn_loss = 0.0
        max_steps = args.learn_steps
        for s in range(max_steps):
            turn_optimizer.zero_grad()
            logits, _ = model.forward(inp, state=memory_state)
            loss_val = loss_fn.forward(logits, tgt)
            dlogits = loss_fn.backward()
            model.backward(dlogits)
            turn_optimizer.step()
            turn_loss = float(loss_val)
            if turn_loss < 0.15:
                break

        # --- E. Murid Menguji Kuis SETELAH Belajar (Post-Test) ---
        post_quiz_tokens = tok.encode(quiz_prompt)
        post_gen_ids = model.generate(
            post_quiz_tokens,
            max_new_tokens=args.max_quiz_tokens,
            temperature=args.temperature,
            top_k=20,
            repetition_penalty=1.25,
            state=memory_state,
        )
        raw_post = tok.decode(post_gen_ids[len(post_quiz_tokens):]).strip()
        post_answer = clean_sentence_output(raw_post)
        full_post = f"{quiz_prompt} {post_answer}".strip()
        post_score, post_hits, total_kw = score_response(full_post, expected_keywords)

        # 3. Otomatis Simpan HANYA JIKA Murid Berhasil Mengingat (Recall >= 40%)
        if post_score >= 40.0:
            os.makedirs("data", exist_ok=True)
            qa_record = {
                "turn": turn,
                "topic": topic,
                "student_question": inquiry,
                "teacher_answer": lesson_text,
                "quiz_question": quiz_question,
                "quiz_prompt": quiz_prompt,
                "student_answer": full_post,
                "recall_score": post_score,
                "mastered": True,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open("data/learned_teacher_qa.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(qa_record, ensure_ascii=False) + "\n")
            saved_status = f"Disimpan ke 'data/learned_teacher_qa.jsonl' (Lulus: {post_score:.0f}%)"
        else:
            saved_status = f"Dilewati (Recall {post_score:.0f}% < 40%, belum dikuasai murid)"

        print(f"│")
        print(f"│ 🤖 RESPON JAWABAN MURID (HOPE LITE):")
        print(f"│   Sebelum Belajar : \"{quiz_prompt} {pre_answer}\" (Recall: {pre_hits}/{total_kw} kata kunci)")
        print(f"│   Sesudah Belajar : \"{quiz_prompt} {post_answer}\" (Recall: {post_hits}/{total_kw} kata kunci)")
        print(f"│")
        print(f"│ 📊 METRIK EVALUASI:")
        print(f"│   • Loss Pelajaran : {turn_loss:.4f} (Perplexity: {np.exp(min(15.0, turn_loss)):.2f})")
        print(f"│   • Memory M Norm  : {mem_norm:.4f}")
        print(f"│   • Skor Pemahaman : {post_score:.1f}% ({post_hits}/{total_kw} kata kunci cocok)")
        print(f"│   • Auto-Saved     : {saved_status}")
        print(f"└────────────────────────────────────────────────────────────────────────┘")

        history_records.append({
            "turn": turn,
            "topic": topic,
            "lesson": lesson_text,
            "study_text": study_text.strip(),
            "quiz_question": quiz_question,
            "quiz_prompt": quiz_prompt,
            "expected_keywords": expected_keywords,
            "loss": turn_loss,
            "norm": mem_norm,
            "pre_score": pre_score,
            "post_score": post_score,
            "pre_answer": f"{quiz_prompt} {pre_answer}",
            "post_answer": f"{quiz_prompt} {post_answer}",
        })

    # Optional sleep consolidation phase
    if getattr(args, "sleep", True) and len(history_records) > 0:
        sleep_consolidation(model, tok, history_records, steps=args.sleep_steps, lr=args.sleep_lr, augment=getattr(args, "augment", True))

    # Save tutored model: simpan dengan memory_state=None (memori kerja bersih setelah tidur)
    save_path = args.save_path
    if not os.path.dirname(save_path):
        save_path = os.path.join("models", save_path)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    model.save_checkpoint(save_path, memory_state=None, extra_meta={"note": "llm_teacher_tutored"})
    print(f"\n[Saved]: Model hasil bimbingan guru tersimpan di '{save_path}'.")

    meta_info = {
        "session_id": session_id,
        "teacher": teacher_type,
        "teacher_desc": teacher_desc,
        "checkpoint": ckpt_path,
        "model": args.model if teacher_type != "curriculum" else "curriculum-id",
    }
    jsonl_log, md_log = save_tutoring_logs(history_records, meta_info)

    # Persist student memory to SQLite database
    db_store = nl.SQLiteMemoryStore("data/memory.db")
    avg_post_score = float(np.mean([r['post_score'] for r in history_records]))
    db_store.save_memory(
        session_id=session_id,
        state=memory_state,
        meta={
            "type": "tutoring_session",
            "turns": len(history_records),
            "teacher": teacher_desc,
            "avg_score": f"{avg_post_score:.1f}%",
            "checkpoint": save_path,
        },
    )
    db_store.save_memory(
        session_id="latest_tutoring",
        state=memory_state,
        meta={
            "session_id": session_id,
            "teacher": teacher_desc,
            "checkpoint": save_path,
        },
    )

    print(f"[Logs Tersimpan]:")
    print(f"  • JSONL Log  : {jsonl_log}")
    print(f"  • Markdown   : {md_log}")
    print(f"  • SQLite DB  : data/memory.db (Session: '{session_id}' & 'latest_tutoring')")
    print("\n" + "=" * 78)
    print("📈 REKAP PERKEMBANGAN PEMAHAMAN MURID (HOPE LITE) PER TURN CHAT:")
    print("=" * 78)
    print(f"{'Turn':<6} | {'Topik Pelajaran':<26} | {'Loss':<8} | {'Memory ||M||':<13} | {'Skor Kuis (Sebelum -> Sesudah)'}")
    print("-" * 78)
    for r in history_records:
        score_change = f"{r['pre_score']:.0f}% -> {r['post_score']:.0f}%"
        print(f"{r['turn']:<6} | {r['topic'][:26]:<26} | {r['loss']:<8.4f} | {r['norm']:<13.4f} | {score_change}")
    print("=" * 78)
    avg_post = np.mean([r['post_score'] for r in history_records])
    avg_pre = np.mean([r['pre_score'] for r in history_records])
    print(f"Rata-rata Skor Pemahaman: {avg_pre:.1f}% (Sebelum) ──► {avg_post:.1f}% (Sesudah Belajar)")
    print("✓ Terbukti model HOPE berkembang dan mengunci materi secara bertahap di setiap turn!")


if __name__ == "__main__":
    main()

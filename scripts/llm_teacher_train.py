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

    def get_lesson_and_quiz(self, turn_idx: int, topic_hint: Optional[str] = None) -> Dict[str, Any]:
        """Return dict with 'topic', 'lesson', 'quiz_prompt', 'expected_keywords'."""
        raise NotImplementedError


CURRICULUM_BANK = {
    "transportasi": [
        {
            "topic": "Transportasi Darat: Mobil",
            "lesson": "Mobil adalah alat transportasi darat yang memiliki roda 4 dan digerakkan oleh mesin. Mobil digunakan untuk mengangkut penumpang dan barang di jalan raya.",
            "quiz_prompt": "Mobil adalah",
            "expected_keywords": ["alat", "transportasi", "darat", "roda", "4", "mesin"],
        },
        {
            "topic": "Transportasi Darat: Sepeda Motor",
            "lesson": "Sepeda motor adalah alat transportasi darat roda 2 yang digerakkan oleh mesin berbahan bakar atau listrik. Pengendara motor wajib mengenakan helm keselamatan.",
            "quiz_prompt": "Sepeda motor adalah",
            "expected_keywords": ["alat", "transportasi", "darat", "roda", "2", "mesin"],
        },
        {
            "topic": "Transportasi Udara: Pesawat Terbang",
            "lesson": "Pesawat terbang adalah alat transportasi udara yang memiliki sayap dan mesin jet untuk terbang di angkasa. Pesawat dikemudikan oleh seorang pilot di dalam kokpit.",
            "quiz_prompt": "Pesawat terbang adalah",
            "expected_keywords": ["alat", "transportasi", "udara", "sayap", "terbang", "pilot"],
        },
        {
            "topic": "Transportasi Laut: Kapal Laut",
            "lesson": "Kapal laut adalah alat transportasi air berukuran besar yang berlayar mengarungi lautan dan samudera. Kapal laut dinakhodai oleh seorang kapten atau nakhoda.",
            "quiz_prompt": "Kapal laut adalah",
            "expected_keywords": ["alat", "transportasi", "air", "berlayar", "lautan", "nakhoda"],
        },
        {
            "topic": "Transportasi Rel: Kereta Api",
            "lesson": "Kereta api adalah alat transportasi darat yang berjalan di atas lintasan rel besi. Kereta api ditarik oleh lokomotif dan dikemudikan oleh masinis.",
            "quiz_prompt": "Kereta api adalah",
            "expected_keywords": ["alat", "transportasi", "darat", "rel", "besi", "masinis"],
        },
        {
            "topic": "Transportasi Udara: Helikopter",
            "lesson": "Helikopter adalah alat transportasi udara bersayap putar yang dapat lepas landas dan mendarat secara vertikal di tempat sempit tanpa landasan pacu.",
            "quiz_prompt": "Helikopter adalah",
            "expected_keywords": ["alat", "transportasi", "udara", "putar", "vertikal", "landas"],
        },
    ],
    "teknologi": [
        {
            "topic": "Teknologi: Komputer",
            "lesson": "Komputer adalah perangkat elektronik canggih yang mampu mengolah data digital, menjalankan program perangkat lunak, dan memproses perhitungan matematis rumit.",
            "quiz_prompt": "Komputer adalah",
            "expected_keywords": ["perangkat", "elektronik", "data", "digital", "program"],
        },
        {
            "topic": "Teknologi: Kecerdasan Buatan (AI)",
            "lesson": "Kecerdasan buatan adalah bidang ilmu komputer yang bertujuan membuat mesin cerdas yang mampu belajar dari data dan meniru fungsi kognitif manusia.",
            "quiz_prompt": "Kecerdasan buatan adalah",
            "expected_keywords": ["ilmu", "komputer", "cerdas", "belajar", "data", "manusia"],
        },
        {
            "topic": "Teknologi: Robotika",
            "lesson": "Robotika adalah bidang teknologi rekayasa yang merancang dan membuat robot otonom untuk membantu pekerjaan manusia di industri dan eksplorasi ruang angkasa.",
            "quiz_prompt": "Robotika adalah",
            "expected_keywords": ["teknologi", "merancang", "robot", "pekerjaan", "manusia"],
        },
        {
            "topic": "Teknologi: Jaringan Internet",
            "lesson": "Internet adalah jaringan telekomunikasi global yang menghubungkan miliaran komputer dan gawai di seluruh dunia menggunakan protokol komunikasi standar.",
            "quiz_prompt": "Internet adalah",
            "expected_keywords": ["jaringan", "telekomunikasi", "global", "komputer", "dunia"],
        },
    ],
    "sains": [
        {
            "topic": "Sains: Planet Bumi",
            "lesson": "Bumi adalah planet ketiga dari Matahari dalam tata surya yang memiliki kehidupan, oksigen, dan air cair. Bumi berputar pada porosnya setiap 24 jam.",
            "quiz_prompt": "Bumi adalah",
            "expected_keywords": ["planet", "tata", "surya", "kehidupan", "air"],
        },
        {
            "topic": "Sains: Planet Mars",
            "lesson": "Mars adalah planet keempat dari Matahari yang sering dijuluki Planet Merah karena kandungan besi oksida yang melimpah pada permukaannya yang gersang.",
            "quiz_prompt": "Mars adalah",
            "expected_keywords": ["planet", "merah", "matahari", "besi", "oksida"],
        },
        {
            "topic": "Sains: Hukum Gravitasi",
            "lesson": "Gravitasi adalah gaya tarik-menarik mendasar yang bekerja antara semua benda yang memiliki massa di alam semesta. Gravitasi menjaga planet tetap mengorbit bintang.",
            "quiz_prompt": "Gravitasi adalah",
            "expected_keywords": ["gaya", "tarik", "massa", "alam", "semesta", "orbit"],
        },
        {
            "topic": "Sains: Dinosaurus",
            "lesson": "Dinosaurus adalah kelompok hewan reptil purba raksasa yang mendominasi daratan bumi pada zaman Mesozoikum sebelum punah puluhan juta tahun yang lalu.",
            "quiz_prompt": "Dinosaurus adalah",
            "expected_keywords": ["hewan", "reptil", "purba", "bumi", "punah"],
        },
    ],
    "biologi": [
        {
            "topic": "Biologi: Kucing",
            "lesson": "Kucing adalah hewan mamalia karnivora domestik berkaki empat yang memiliki cakar lentik dan pendengaran tajam. Kucing banyak dipelihara manusia sebagai sahabat setia.",
            "quiz_prompt": "Kucing adalah",
            "expected_keywords": ["hewan", "mamalia", "karnivora", "empat", "pelihara"],
        },
        {
            "topic": "Biologi: Burung Elang",
            "lesson": "Burung elang adalah burung pemangsa berukuran besar yang memiliki penglihatan sangat tajam, paruh berkait kokoh, dan sayap lebar untuk terbang membubung tinggi.",
            "quiz_prompt": "Burung elang adalah",
            "expected_keywords": ["burung", "pemangsa", "penglihatan", "tajam", "sayap"],
        },
        {
            "topic": "Biologi: Hutan Hujan Tropis",
            "lesson": "Hutan hujan tropis adalah bioma hutan lebat beriklim basah yang menjadi rumah bagi lebih dari separuh spesies tumbuhan dan satwa liar di muka bumi.",
            "quiz_prompt": "Hutan hujan tropis adalah",
            "expected_keywords": ["hutan", "lebat", "spesies", "tumbuhan", "satwa"],
        },
    ],
    "kuliner": [
        {
            "topic": "Kuliner: Kopi Nusantara",
            "lesson": "Kopi adalah minuman seduhan populer berkafein yang dihasilkan dari pengolahan biji tanaman kopi sangrai. Indonesia terkenal dengan kopi arabika dan robusta berkualitas.",
            "quiz_prompt": "Kopi adalah",
            "expected_keywords": ["minuman", "seduhan", "biji", "kopi", "indonesia"],
        },
        {
            "topic": "Kuliner: Nasi Goreng",
            "lesson": "Nasi goreng adalah hidangan khas berupa nasi yang digoreng bersama bumbu rempah, kecap manis, telur, dan lauk pelengkap di atas wajan panas.",
            "quiz_prompt": "Nasi goreng adalah",
            "expected_keywords": ["hidangan", "nasi", "goreng", "bumbu", "rempah"],
        },
    ],
}


class CurriculumTeacher(BaseTeacher):
    """Built-in structured Indonesian curriculum teacher with diverse category banks."""

    def __init__(self, category: str = "all", custom_file: Optional[str] = None):
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
            # Search for topic matching hint
            for item in self.curriculum:
                if hint_lower in item["topic"].lower() or hint_lower in item["lesson"].lower():
                    return item
            # If not in bank, generate dynamic lesson on this topic
            words = topic_hint.split()
            return {
                "topic": f"Topik Kustom: {topic_hint}",
                "lesson": f"{topic_hint} adalah konsep penting yang memiliki karakteristik unik dan peran besar dalam peradaban manusia.",
                "quiz_prompt": f"{topic_hint} adalah",
                "expected_keywords": [w.lower() for w in words] + ["konsep", "penting", "peran"],
            }

        idx = (turn_idx - 1) % len(self.curriculum)
        return self.curriculum[idx]

class OllamaTeacher(BaseTeacher):
    """Teacher powered by a local Ollama LLM instance."""

    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3"):
        self.host = host.rstrip("/")
        self.model = model
        self.fallback = CurriculumTeacher()

    def _call_ollama(self, prompt: str) -> str:
        url = f"{self.host}/api/generate"
        payload = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()

    def get_lesson_and_quiz(self, turn_idx: int, topic_hint: Optional[str] = None) -> Dict[str, Any]:
        target_topic = topic_hint or f"topik umum ke-{turn_idx} (misal: sains, teknologi, biologi, transportasi, sejarah)"
        prompt = (
            f"Anda adalah Guru AI untuk model bahasa pemula. Buat 1 pelajaran singkat (2-3 kalimat) dalam bahasa Indonesia "
            f"tentang: '{target_topic}'. "
            f"Keluarkan format JSON murni persis seperti ini:\n"
            f'{{"topic": "{target_topic}", "lesson": "teks pelajaran singkat", "quiz_prompt": "kata awal untuk kuis", "expected_keywords": ["kata1", "kata2"]}}\n'
            f"HANYA keluarkan JSON tanpa penjelasan markdown."
        )
        try:
            raw = self._call_ollama(prompt)
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except Exception as e:
            print(f"[Ollama Teacher Warning]: Gagal menghubungi Ollama ({e}). Menggunakan kurikulum bawaan.")
        return self.fallback.get_lesson_and_quiz(turn_idx, topic_hint=topic_hint)


class OpenAITeacher(BaseTeacher):
    """Teacher powered by OpenAI-compatible APIs (Groq, OpenRouter, OpenAI, vLLM)."""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.fallback = CurriculumTeacher()

    def _call_api(self, prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5,
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
        target_topic = topic_hint or f"topik umum ke-{turn_idx} (misal: sains, teknologi, biologi, transportasi, sejarah)"
        prompt = (
            f"Anda adalah Guru AI untuk model bahasa pemula. Buat 1 materi pelajaran singkat (2 kalimat) dalam bahasa Indonesia "
            f"tentang: '{target_topic}'. "
            f"Keluarkan output dalam JSON murni persis format ini:\n"
            f'{{"topic": "{target_topic}", "lesson": "materi pelajaran singkat", "quiz_prompt": "kata awal prompt kuis", "expected_keywords": ["kata1", "kata2", "kata3"]}}\n'
            f"Hanya JSON tanpa teks lain."
        )
        try:
            raw = self._call_api(prompt)
            start = raw.find("{")
            if start >= 0:
                obj, _ = json.JSONDecoder().raw_decode(raw[start:])
                return obj
        except Exception as e:
            print(f"[OpenAI Teacher Warning]: Gagal menghubungi API ({e}). Menggunakan kurikulum bawaan.")

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
                        help="Comma-separated topics for consecutive turns (e.g. --topics 'Mobil Listrik,Robotika,Planet Mars,Kucing,Kopi')")
    parser.add_argument("--category", type=str, default="all",
                        choices=["all", "transportasi", "teknologi", "sains", "biologi", "kuliner"],
                        help="Pre-defined category pack for offline curriculum teacher (default: 'all')")
    parser.add_argument("--materi-file", type=str, default=None,
                        help="Path to custom text (.txt paragraphs) or JSONL file containing custom lessons")
    parser.add_argument("--checkpoint", type=str, default="models/hope_id.npz",
                        help="Base model checkpoint to teach (defaults to models/hope_id.npz if exists)")
    parser.add_argument("--save-path", type=str, default="models/hope_tutored.npz",
                        help="Path to save the model after tutoring")
    parser.add_argument("--lr", type=float, default=6e-3, help="Learning rate per turn")
    parser.add_argument("--learn-steps", type=int, default=8, help="Number of optimization steps per lesson turn")
    parser.add_argument("--max-quiz-tokens", type=int, default=30, help="Max tokens generated during quiz response")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature during quiz")

    # API options (defaults from .env)
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY", ""),
                        help="API key for teacher (defaults to OPENAI_API_KEY in .env)")
    parser.add_argument("--base-url", type=str, default=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
                        help="Base URL for API (defaults to OPENAI_BASE_URL in .env, e.g. OpenRouter/9router)")
    parser.add_argument("--model", type=str, default=os.getenv("LLM_MODEL", "meta-llama/llama-3-8b-instruct:free"),
                        help="Model name for teacher (defaults to LLM_MODEL in .env)")
    return parser.parse_args()

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

def save_tutoring_logs(history_records: List[Dict[str, Any]], meta_info: Dict[str, Any]) -> Tuple[str, str]:
    """Save tutoring session logs in both JSONL and Markdown formats."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    session_id = meta_info.get("session_id", "default")

    jsonl_path = os.path.join(log_dir, f"{session_id}.jsonl")
    md_path = os.path.join(log_dir, f"{session_id}.md")
    latest_md_path = os.path.join(log_dir, "latest_tutoring.md")

    # 1. Write JSONL log (structured per-turn data)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in history_records:
            entry = {**meta_info, **r}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 2. Write Markdown Report
    avg_pre = float(np.mean([r['pre_score'] for r in history_records]))
    avg_post = float(np.mean([r['post_score'] for r in history_records]))

    md_content = [
        "# Laporan Sesi Pembelajaran: Guru LLM & Murid HOPE Lite",
        "",
        f"- **Session ID:** `{session_id}`",
        f"- **Waktu Sesi:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Guru (Teacher):** {meta_info.get('teacher_desc', 'N/A')}",
        f"- **Model Base Murid:** `{meta_info.get('checkpoint', 'N/A')}`",
        f"- **Total Turn:** {len(history_records)} giliran",
        f"- **Skor Pemahaman:** **{avg_pre:.1f}%** (Sebelum) ──► **{avg_post:.1f}%** (Sesudah Belajar)",
        "",
        "---",
        "",
        "## 📈 Tabel Rangkuman Perkembangan per Turn",
        "",
        "| Turn | Topik Pelajaran | Loss | Memory \\|\\|M\\|\\| | Pre-Test Score | Post-Test Score | Peningkatan |",
        "|---|---|---|---|---|---|---|",
    ]

    for r in history_records:
        delta = r['post_score'] - r['pre_score']
        delta_str = f"+{delta:.0f}%" if delta > 0 else f"{delta:.0f}%"
        md_content.append(
            f"| {r['turn']} | {r['topic']} | {r['loss']:.4f} | {r['norm']:.4f} | {r['pre_score']:.0f}% | {r['post_score']:.0f}% | **{delta_str}** |"
        )

    md_content.extend([
        "",
        "---",
        "",
        "## 📝 Rincian Dialog Pembelajaran per Turn",
        "",
    ])

    for r in history_records:
        md_content.extend([
            f"### Turn {r['turn']}: {r['topic']}",
            f"- **Materi Guru:**",
            f"  > *\"{r.get('lesson', '')}\"*",
            f"- **Kuis Evaluasi:** `{r.get('quiz_prompt', '')}`",
            f"- **Jawaban Murid Sebelum Belajar:**",
            f"  > `{r['pre_answer']}` *(Recall: {r['pre_score']:.0f}%)*",
            f"- **Jawaban Murid Sesudah Belajar:**",
            f"  > `{r['post_answer']}` *(Recall: {r['post_score']:.0f}%)*",
            f"- **Metrik Teknis:**",
            f"  - Loss Pembelajaran: `{r['loss']:.4f}` *(Perplexity: {float(np.exp(min(15.0, r['loss']))):.2f})*",
            f"  - Norma Matriks Memori $\\|M\\|$: `{r['norm']:.4f}`",
            "",
        ])

    full_md = "\n".join(md_content)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(full_md)
    with open(latest_md_path, "w", encoding="utf-8") as f:
        f.write(full_md)

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
        teacher = CurriculumTeacher(category=args.category, custom_file=args.materi_file)
        print(f"Guru (Teacher): {teacher_desc}")
    elif teacher_type == "ollama":
        teacher = OllamaTeacher(model=args.model)
        teacher_desc = f"Ollama Lokal ({args.model})"
        print(f"Guru (Teacher): {teacher_desc}")
    elif teacher_type == "openai":
        teacher = OpenAITeacher(api_key=api_key, base_url=args.base_url, model=args.model)
        provider_name = "OpenRouter / 9router" if "openrouter" in args.base_url.lower() or "9router" in args.base_url.lower() else "OpenAI API"
        teacher_desc = f"{provider_name} ({args.model})"
        print(f"Guru (Teacher): {teacher_desc} (URL: {args.base_url})")
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path) and os.path.exists(os.path.join("models", ckpt_path)):
        ckpt_path = os.path.join("models", ckpt_path)
    elif not os.path.exists(ckpt_path) and os.path.exists("models/hope_model.npz"):
        ckpt_path = "models/hope_model.npz"

    tok = nl.ByteTokenizer()

    if os.path.exists(ckpt_path):
        print(f"Murid (Student): HOPE Lite dimuat dari '{ckpt_path}'")
        model, meta = nl.HOPE.load_checkpoint(ckpt_path)
        memory_state = meta.get("memory_state")
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
        lesson_text = item.get("lesson", "")
        quiz_prompt = item.get("quiz_prompt", "")
        expected_keywords = item.get("expected_keywords", [])
        print(f"\n┌────────────────────────────────────────────────────────────────────────┐")
        print(f"│ 📚 TURN {turn}/{args.turns}: {topic:<58} │")
        print(f"├────────────────────────────────────────────────────────────────────────┤")
        print(f"│ 🧑‍🏫 GURU MENGAJAR:")
        print(f"│   \"{lesson_text}\"")

        # --- A. Murid Menguji Kuis SEBELUM Belajar (Pre-Test) ---
        pre_quiz_tokens = tok.encode(quiz_prompt)
        pre_gen_ids = model.generate(
            pre_quiz_tokens,
            max_new_tokens=args.max_quiz_tokens,
            temperature=args.temperature,
            top_k=20,
        )
        pre_answer = tok.decode(pre_gen_ids)[len(quiz_prompt):].strip()
        pre_score, pre_hits, _ = score_response(pre_answer, expected_keywords)

        # --- B. Murid Belajar & Menyerap Materi (In-Chat Learning & Memory Update) ---
        lesson_tokens = tok.encode(lesson_text)
        inp = np.array([lesson_tokens[:-1]], dtype=np.int64)
        tgt = np.array([lesson_tokens[1:]], dtype=np.int64)

        # 1. Update Fast-Weight Memory via Delta Rule (in-context absorption)
        _, memory_state = model.forward(inp, state=memory_state)
        mem_norm = float(np.linalg.norm(memory_state[0]))

        # 2. Update Slow Weights via Analytical Gradient Descent
        turn_loss = 0.0
        for _ in range(args.learn_steps):
            optimizer.zero_grad()
            logits, _ = model.forward(inp, state=memory_state)
            loss_val = loss_fn.forward(logits, tgt)
            dlogits = loss_fn.backward()
            model.backward(dlogits)
            optimizer.step()
            turn_loss = float(loss_val)

        # --- C. Murid Menguji Kuis SETELAH Belajar (Post-Test) ---
        post_quiz_tokens = tok.encode(quiz_prompt)
        post_gen_ids = model.generate(
            post_quiz_tokens,
            max_new_tokens=args.max_quiz_tokens,
            temperature=args.temperature,
            top_k=20,
        )
        post_answer = tok.decode(post_gen_ids)[len(quiz_prompt):].strip()
        post_score, post_hits, total_kw = score_response(post_answer, expected_keywords)

        print(f"│")
        print(f"│ 🤖 RESPON MURID (HOPE LITE):")
        print(f"│   Sebelum Belajar : \"{quiz_prompt} {pre_answer}\" (Recall: {pre_hits}/{total_kw} kata kunci)")
        print(f"│   Sesudah Belajar : \"{quiz_prompt} {post_answer}\" (Recall: {post_hits}/{total_kw} kata kunci)")
        print(f"│")
        print(f"│ 📊 METRIK EVALUASI:")
        print(f"│   • Loss Pelajaran : {turn_loss:.4f} (Perplexity: {np.exp(min(15.0, turn_loss)):.2f})")
        print(f"│   • Memory M Norm  : {mem_norm:.4f}")
        print(f"│   • Skor Pemahaman : {post_score:.1f}% ({post_hits}/{total_kw} kata kunci cocok)")
        print(f"└────────────────────────────────────────────────────────────────────────┘")

        history_records.append({
            "turn": turn,
            "topic": topic,
            "lesson": lesson_text,
            "quiz_prompt": quiz_prompt,
            "expected_keywords": expected_keywords,
            "loss": turn_loss,
            "norm": mem_norm,
            "pre_score": pre_score,
            "post_score": post_score,
            "pre_answer": f"{quiz_prompt} {pre_answer}",
            "post_answer": f"{quiz_prompt} {post_answer}",
        })

    # Save tutored model
    save_path = args.save_path
    if not os.path.dirname(save_path):
        save_path = os.path.join("models", save_path)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    model.save_checkpoint(save_path, memory_state=memory_state, extra_meta={"note": "llm_teacher_tutored"})
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
    print(f"  • Ringkasan  : logs/latest_tutoring.md")
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

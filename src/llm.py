"""LLM answer synthesis layer with graceful degradation.

Detects which local/cloud LLM backend is available (Ollama, then Gemini),
and exposes a single synthesize() entry point that returns natural-language
answers grounded in retrieved document fragments.

If no backend is reachable the module returns None — the UI must remain
functional with retrieval results alone.
"""

import logging
import os

import requests

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"
SYSTEM_PROMPT = (
    "Eres un asistente legal especializado en la legislación "
    "laboral mexicana. Responde en español, de forma clara y "
    "concisa. Basa tu respuesta ÚNICAMENTE en los fragmentos "
    "proporcionados. "
    "INSTRUCCIÓN OBLIGATORIA: Al final de tu respuesta, incluye "
    "una sección titulada 'Fuente principal:' donde cites "
    "textualmente el fragmento más relevante del documento más "
    "relevante (máximo 2 oraciones). Usa este formato exacto:\n"
    "Fuente principal: [título del documento]\n"
    "'[cita textual del fragmento más relevante]'\n"
    "Si la información no está en los fragmentos, indícalo "
    "explícitamente. No inventes artículos ni números de ley."
)


def detect_llm() -> dict:
    """Detect the best available LLM backend at startup.

    Probes Ollama first (local, free, fast), then falls back to Gemini if
    GEMINI_API_KEY is set. Returns a status dict that the UI can render.
    """
    # ── Try Ollama ───────────────────────────────────────────────────────
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if r.status_code == 200:
            tags = r.json().get("models", [])
            available_names = [m.get("name", "") for m in tags]

            try:
                import psutil
                ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            except Exception:
                ram_gb = 0.0

            if ram_gb >= 15:
                preferred = "qwen2.5:7b"
            elif ram_gb >= 7:
                preferred = "llama3.2:3b"
            else:
                preferred = "phi3:mini"

            if preferred in available_names:
                model = preferred
            elif available_names:
                model = available_names[0]
                logger.info("Preferred model %s not available — using %s",
                            preferred, model)
            else:
                model = None

            if model:
                return {
                    "provider": "ollama",
                    "model": model,
                    "available": True,
                    "reason": f"Ollama reachable, using {model}",
                }
    except Exception as e:
        logger.info("Ollama not reachable: %s", e)

    # ── Try Gemini ───────────────────────────────────────────────────────
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        return {
            "provider": "gemini",
            "model": "gemini-1.5-flash",
            "available": True,
            "reason": "Gemini API key detected",
        }

    return {
        "provider": "none",
        "model": None,
        "available": False,
        "reason": "No LLM available. Install Ollama or set GEMINI_API_KEY.",
    }


def get_document_text(pdf_path: str, query_tokens: list = None, max_chars: int = 800) -> str:
    """Return relevant text from a PDF; finds best passage when query_tokens provided."""
    try:
        from src.pdf_extractor import extract_text
        text = extract_text(pdf_path)
        if not text:
            return ""
        if not query_tokens:
            return text[:max_chars]
        tokens_set = set(t.lower() for t in query_tokens)
        best_start = 0
        best_score = -1
        step = max(1, max_chars // 2)
        for start in range(0, len(text), step):
            chunk = text[start:start + max_chars]
            score = sum(1 for tok in tokens_set if tok in chunk.lower())
            if score > best_score:
                best_score = score
                best_start = start
        return text[best_start:best_start + max_chars].strip()
    except Exception as e:
        logger.warning("Could not extract text from %s: %s", pdf_path, e)
        return ""


def _build_prompt(query: str, results: list, query_tokens: list = None) -> str:
    """Compose the user prompt with the retrieved fragments inlined."""
    fragments_blocks = []
    for r in results:
        url = r.get("url") or f"{r.get('title', 'unknown')}.pdf"
        # If url is a bare filename, look under data/pdfs/
        if url and not os.path.isabs(url) and not url.startswith("data/"):
            url_full = os.path.join("data", "pdfs", url)
        else:
            url_full = url
        text = get_document_text(url_full, query_tokens=query_tokens, max_chars=800)
        fragments_blocks.append(f"--- {r.get('title', 'doc')} ---\n{text}")

    fragments = "\n\n".join(fragments_blocks) if fragments_blocks else "(sin fragmentos)"
    return (
        f"Pregunta: {query}\n\n"
        f"Fragmentos relevantes de la Ley Federal del Trabajo:\n"
        f"{fragments}\n\n"
        f"Responde la pregunta basándote en los fragmentos anteriores."
    )


def _call_ollama(model: str, prompt: str) -> str | None:
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": full_prompt, "stream": False},
            timeout=60,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip() or None
    except Exception as e:
        logger.warning("Ollama call failed: %s", e)
        return None


def _call_gemini(prompt: str) -> str | None:
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            return None
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            system_instruction=SYSTEM_PROMPT,
        )
        response = model.generate_content(prompt)
        return (response.text or "").strip() or None
    except Exception as e:
        logger.warning("Gemini call failed: %s", e)
        return None


def get_available_models() -> dict:
    """Returns all available LLM options for manual selection.

    Returns a dict with 'options' (display strings), 'values' (provider/model
    tuples), and 'default_index' (auto-detected best option).
    """
    options = []
    values = []

    options.append("Desactivado — solo recuperación")
    values.append(('none', None))

    try:
        import requests as _req
        resp = _req.get("http://localhost:11434/api/tags", timeout=2)
        if resp.status_code == 200:
            tags = resp.json().get('models', [])
            for m in tags:
                name = m.get('name', '')
                if name:
                    options.append(f"Local — {name}")
                    values.append(('ollama', name))
    except Exception:
        pass

    if os.getenv('GEMINI_API_KEY'):
        options.append("Gemini 1.5 Flash (API)")
        values.append(('gemini', 'gemini-1.5-flash'))

    auto = detect_llm()
    default_index = 0
    for i, (provider, model) in enumerate(values):
        if provider == auto['provider'] and model == auto['model']:
            default_index = i
            break

    return {
        'options': options,
        'values': values,
        'default_index': default_index
    }


def synthesize(query: str, results: list[dict],
               provider: str, model: str | None,
               query_tokens: list = None) -> str | None:
    """Generate a natural-language answer grounded in *results*.

    Returns None when provider is 'none' or the call fails — the caller
    must keep the UI functional regardless.
    """
    if provider == 'none':
        return None

    # Build user prompt with all retrieved documents; mark the most relevant
    fragments_text = ""
    for i, r in enumerate(results or []):
        url = r.get("url") or f"{r.get('title', 'unknown')}.pdf"
        if url and not os.path.isabs(url) and not url.startswith("data/"):
            pdf_path = os.path.join("data", "pdfs", url)
        else:
            pdf_path = url
        text = get_document_text(pdf_path, query_tokens=query_tokens, max_chars=800)
        if not text:
            continue
        marker = " [DOCUMENTO MÁS RELEVANTE]" if i == 0 else ""
        fragments_text += (
            f"\n--- {r.get('title', 'doc')}{marker} "
            f"(similitud: {r.get('score', 0):.4f}) ---\n{text}\n"
        )

    if not fragments_text:
        fragments_text = "(sin fragmentos)"

    prompt = (
        f"Pregunta: {query}\n\n"
        f"Fragmentos de la legislación laboral mexicana:\n"
        f"{fragments_text}\n\n"
        "Responde la pregunta y al final cita el fragmento más "
        "relevante como se indicó en las instrucciones."
    )

    if provider == "ollama":
        return _call_ollama(model, prompt)
    if provider == "gemini":
        return _call_gemini(prompt)
    return None

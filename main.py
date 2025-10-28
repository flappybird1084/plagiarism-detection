from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from flask import (
    abort,
    flash,
    redirect,
    render_template,
    request,
    url_for,
    Flask,
    jsonify,
)
from flask_sqlalchemy import SQLAlchemy
from html.parser import HTMLParser
from urllib.parse import urlparse

import requests

from checker import checked


app = Flask(__name__)
app.secret_key = "dev"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///plagiarism.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_GENERATE_MODEL = "gpt-oss:latest"


class Document(db.Model):
    __tablename__ = "documents"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=False)
    is_source = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<Document id={self.id} title={self.title!r} is_source={self.is_source}>"


def _decode_upload(upload) -> str:
    """Return a UTF-8 string from an uploaded file."""
    payload = upload.read()
    if not payload:
        return ""
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload.decode("utf-8", errors="ignore")


def _fetch_selected_sources(selected_ids: Iterable[int]) -> List[Document]:
    if not selected_ids:
        return []
    return (
        Document.query.filter(Document.id.in_(selected_ids), Document.is_source.is_(True))
        .order_by(Document.created_at.desc())
        .all()
    )


class _HTMLTextExtractor(HTMLParser):
    """Extract readable text and title from an HTML document."""

    _block_elements = {
        "p",
        "div",
        "section",
        "article",
        "header",
        "footer",
        "li",
        "ul",
        "ol",
        "br",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }

    def __init__(self) -> None:
        super().__init__()
        self._parts: List[str] = []
        self._title_parts: List[str] = []
        self._skip_depth = 0
        self._collecting_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style"}:
            self._skip_depth += 1
            return
        if lowered == "title":
            self._collecting_title = True
            return
        if lowered in self._block_elements:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style"} and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if lowered == "title":
            self._collecting_title = False
            return
        if lowered in self._block_elements:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        cleaned = data.strip()
        if not cleaned:
            return
        if self._collecting_title:
            self._title_parts.append(cleaned)
        else:
            self._parts.append(cleaned + " ")

    @property
    def title_text(self) -> str | None:
        title = " ".join(self._title_parts).strip()
        return title or None

    def get_text(self) -> str:
        text = "".join(self._parts)
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)


def _normalize_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    if not parsed.scheme:
        raw_url = f"https://{raw_url}"
        parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Provide a valid http or https URL.")
    return raw_url


def _fetch_url_content(url: str) -> tuple[str, str | None]:
    try:
        response = requests.get(url, timeout=10)
    except requests.RequestException as exc:
        raise ValueError(f"Unable to fetch URL: {exc}") from exc

    if response.status_code != 200:
        raise ValueError(f"URL returned status code {response.status_code}.")

    content_type = response.headers.get("Content-Type", "")
    text_payload = response.text
    if "text/html" in content_type.lower() or "<html" in text_payload[:200].lower():
        extractor = _HTMLTextExtractor()
        extractor.feed(text_payload)
        extractor.close()
        extracted_text = extractor.get_text().strip()
        if not extracted_text:
            raise ValueError("Fetched page does not contain readable text.")
        return extracted_text, extractor.title_text or response.url

    if content_type and not content_type.lower().startswith("text/"):
        raise ValueError("Only text and HTML content can be saved from URLs.")

    stripped = text_payload.strip()
    if not stripped:
        raise ValueError("Fetched page did not contain any text.")
    return stripped, response.url


def _normalize_base_url(raw_url: str) -> str:
    base = raw_url.strip()
    if not base:
        return OLLAMA_BASE_URL
    if not base.startswith(("http://", "https://")):
        base = f"http://{base}"
    return base


def _generate_source_content(title: str, topic: str, base_url: str | None = None) -> str:
    prompt = (
        "You are generating a detailed reference document that can be used as a source "
        "for plagiarism detection exercises.\n\n"
        f"Topic: {topic}\n"
        f"Title: {title}\n\n"
        "Write a well-structured article with multiple paragraphs, clear sections, "
        "and as much factual detail as possible. Aim for 400-600 words. Avoid markdown "
        "headings or bullet lists; respond with plain text only."
    )
    payload = {
        "model": OLLAMA_GENERATE_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    resolved_base = _normalize_base_url(base_url or OLLAMA_BASE_URL)
    endpoint = f"{resolved_base.rstrip('/')}/api/generate"

    try:
        response = requests.post(endpoint, json=payload, timeout=180)
    except requests.RequestException as exc:
        raise ValueError(f"Ollama request failed: {exc}") from exc

    if response.status_code != 200:
        raise ValueError(
            f"Ollama generate endpoint returned {response.status_code}: {response.text}"
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise ValueError(f"Unable to parse Ollama response: {exc}") from exc

    if body.get("error"):
        raise ValueError(f"Ollama reported an error: {body['error']}")

    text = body.get("response", "")
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Ollama response did not include generated text.")
    return cleaned


@app.route("/", methods=["GET"])
def index():
    sources = (
        Document.query.filter_by(is_source=True)
        .order_by(Document.created_at.desc())
        .all()
    )
    return render_template("index.html", source_documents=sources)


@app.route("/sources", methods=["GET"])
def list_sources():
    sources = (
        Document.query.filter_by(is_source=True)
        .order_by(Document.created_at.desc())
        .all()
    )
    return render_template("sources.html", source_documents=sources)


@app.route("/sources/<int:document_id>/edit", methods=["GET", "POST"])
def edit_source(document_id: int):
    document = Document.query.filter_by(id=document_id, is_source=True).first()
    if document is None:
        abort(404)

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        content = request.form.get("content", "").strip()

        if not content:
            flash("Source document must have content.")
            return redirect(url_for("edit_source", document_id=document_id))

        document.title = title or document.title
        document.content = content
        db.session.commit()
        flash(f"Updated source document '{document.title}'.")
        return redirect(url_for("list_sources"))

    return render_template("edit_source.html", document=document)


@app.route("/sources/<int:document_id>/delete", methods=["POST"])
def delete_source(document_id: int):
    document = Document.query.filter_by(id=document_id, is_source=True).first()
    if document is None:
        flash("Source document not found.")
        return redirect(url_for("list_sources"))

    db.session.delete(document)
    db.session.commit()
    flash(f"Deleted source document '{document.title}'.")
    return redirect(url_for("list_sources"))


@app.route("/add-source", methods=["POST"])
def add_source():
    title = request.form.get("source_title", "").strip()
    typed_text = request.form.get("source_text", "").strip()
    upload = request.files.get("source_file")
    source_url_input = request.form.get("source_url", "").strip()
    normalized_url = ""
    url_inferred_title: str | None = None

    content = ""
    if source_url_input:
        try:
            normalized_url = _normalize_url(source_url_input)
        except ValueError as exc:
            flash(str(exc))
            return redirect(url_for("index"))
        try:
            content, url_inferred_title = _fetch_url_content(normalized_url)
        except ValueError as exc:
            flash(str(exc))
            return redirect(url_for("index"))
    elif upload and upload.filename:
        content = _decode_upload(upload)
        if not title:
            title = Path(upload.filename).name
    elif typed_text:
        content = typed_text

    if not content:
        flash("Provide a URL, file, or typed text to add a source document.")
        return redirect(url_for("index"))

    resolved_title = title or url_inferred_title or normalized_url or "Unnamed Source"
    db.session.add(
        Document(title=resolved_title, content=content, is_source=True)
    )
    db.session.commit()

    flash(f"Added source document '{resolved_title}'.")
    return redirect(url_for("index"))


@app.route("/generate-source", methods=["POST"])
def generate_source():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    topic = (data.get("topic") or "").strip()
    base_url = (data.get("base_url") or "").strip()

    if not topic:
        return jsonify({"error": "Provide a topic to generate a source document."}), 400

    resolved_title = title or f"AI Source: {topic}"
    try:
        generated_content = _generate_source_content(
            resolved_title, topic, base_url=base_url or None
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 502

    document = Document(title=resolved_title, content=generated_content, is_source=True)
    db.session.add(document)
    db.session.commit()

    return jsonify(
        {
            "message": f"Generated source document '{resolved_title}'.",
            "redirect_url": url_for("list_sources"),
            "document": {
                "id": document.id,
                "title": document.title,
            },
        }
    )


@app.route("/check", methods=["POST"])
def check_submission():
    selected_ids: List[int] = []
    for raw_id in request.form.getlist("source_selection"):
        try:
            selected_ids.append(int(raw_id))
        except ValueError:
            continue

    threshold_raw = request.form.get("similarity_threshold", "0.3")
    try:
        similarity_threshold = float(threshold_raw)
    except ValueError:
        similarity_threshold = 0.3
    similarity_threshold = max(0.0, min(1.0, similarity_threshold))

    student_title = request.form.get("student_title", "").strip()
    student_text = request.form.get("student_text", "").strip()
    student_upload = request.files.get("student_file")

    student_content = ""
    if student_upload and student_upload.filename:
        student_content = _decode_upload(student_upload)
        if not student_title:
            student_title = Path(student_upload.filename).name
    elif student_text:
        student_content = student_text

    if not student_content:
        flash("Upload or type a student document before checking.")
        return redirect(url_for("index"))

    selected_sources = _fetch_selected_sources(selected_ids)
    if not selected_sources:
        flash("Choose at least one valid source document to compare against.")
        return redirect(url_for("index"))

    resolved_student_title = student_title or "Student submission"
    student_document = Document(
        title=resolved_student_title, content=student_content, is_source=False
    )
    db.session.add(student_document)
    db.session.commit()

    source_payload = [
        {"id": doc.id, "title": doc.title, "content": doc.content}
        for doc in selected_sources
    ]
    analysis = checked.analyze(student_content, [doc["content"] for doc in source_payload])
    per_source_scores = checked.per_source_scores(
        student_content, [doc["content"] for doc in source_payload]
    )
    student_sentences = checked.split_sentences(student_content)
    highlights = {
        str(doc["id"]): checked.sentence_similarities(student_content, doc["content"])
        for doc in source_payload
    }

    return render_template(
        "results.html",
        scores=analysis,
        student={
            "title": resolved_student_title,
            "content": student_content,
            "sentences": student_sentences,
        },
        sources=[
            {
                **doc,
                "scores": per_source_scores[idx] if idx < len(per_source_scores) else {},
            }
            for idx, doc in enumerate(source_payload)
        ],
        highlights=highlights,
        threshold=similarity_threshold,
    )


def main():
    with app.app_context():
        db.create_all()
    app.run(port=5001, debug=True)


if __name__ == "__main__":
    main()

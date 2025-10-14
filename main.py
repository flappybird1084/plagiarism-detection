from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from flask import abort, flash, redirect, render_template, request, url_for, Flask
from flask_sqlalchemy import SQLAlchemy

from checker import checked


app = Flask(__name__)
app.secret_key = "dev"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///plagiarism.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


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


@app.route("/add-source", methods=["POST"])
def add_source():
    title = request.form.get("source_title", "").strip()
    typed_text = request.form.get("source_text", "").strip()
    upload = request.files.get("source_file")

    content = ""
    if upload and upload.filename:
        content = _decode_upload(upload)
        if not title:
            title = Path(upload.filename).name
    elif typed_text:
        content = typed_text

    if not content:
        flash("Provide a file or typed text to add a source document.")
        return redirect(url_for("index"))

    resolved_title = title or "Unnamed Source"
    db.session.add(
        Document(title=resolved_title, content=content, is_source=True)
    )
    db.session.commit()

    flash(f"Added source document '{resolved_title}'.")
    return redirect(url_for("index"))


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

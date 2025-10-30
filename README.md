# Plagiarism Detection Web App

An interactive Flask application for comparing student submissions against a library of source documents. The tool combines token overlap, TF-IDF similarity, and Ollama-powered embeddings to highlight potential plagiarism and provide per-source similarity scores.

---

## Features
- Upload source documents manually, via file upload, or by fetching a URL.
- Generate new reference material with an Ollama text-generation model.
- Run plagiarism checks using TF-IDF and embedding-based similarity metrics.
- Inspect sentence-level highlights in a browser-based report.

---

## Prerequisites
- **Python**: 3.10 or newer.
- **uv** *(recommended)*: for dependency management and running tasks. Install via `pip install uv`.
- **pip** *(alternative)*: standard Python package installer if you prefer not to use uv.
- **Ollama**: running locally with access to an embedding model (`nomic-embed-text`) and a generation model (`gpt-oss:latest` by default).
  - Download from [https://ollama.ai](https://ollama.ai) and ensure the service is running on `http://localhost:11434`.

---

## Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd plagiarism
   ```

2. **Install Dependencies**
   - With **uv** (recommended):
     ```bash
     uv sync
     ```
   - With **pip**:
     ```bash
     python -m venv .venv
     source .venv/bin/activate  # Windows: .venv\Scripts\activate
     pip install flask flask-sqlalchemy scikit-learn requests
     ```

---

## Running the Application

1. **Start Ollama**
   Ensure the Ollama service is running and that the required models are available:
   ```bash
   ollama pull nomic-embed-text
   ollama pull gpt-oss:latest
   ```

2. **Launch the Flask App**
   ```bash
   uv run main.py
   ```
   - The server starts on `http://localhost:5001`.
   - On first run, a SQLite database (`plagiarism.db`) is created under the `instance/` directory.

3. **Open the Interface**
   Visit `http://localhost:5001` in your browser to begin adding source material and student submissions.

---

## Using the Web Interface

1. **Add Source Documents**
   - **URL**: Paste a link to fetch readable text from a webpage.
   - **File Upload**: Submit plaintext or HTML files.
   - **Manual Entry**: Type or paste content directly into the form.
   - **AI Generation**: Provide a topic (and optional title/base URL) to have Ollama generate a source document.

2. **Check a Student Submission**
   - Upload a file or paste text for the student work.
   - Select one or more source documents to compare against.
   - Optionally adjust the similarity threshold displayed in the results.
   - Submit the form to view detailed similarity metrics and sentence-level highlights.

3. **Interpret the Results**
   - Review overall token, sentence, and paragraph similarity scores.
   - Inspect per-source breakdowns to identify which references contributed to flagged similarities.
   - Use the highlights to see which sentences may need closer review.

---

## Configuration Tips

- **Changing the Ollama Endpoint**: Supply an alternate base URL in the form when running a check (e.g., if Ollama is hosted on another machine).
- **Database Management**: The application uses SQLite by default. Delete `instance/plagiarism.db` to reset the database.
- **Debug Mode**: The app runs in debug mode by default (`app.run(..., debug=True)`). Disable for production use.

---

## Troubleshooting

- **Ollama Connection Errors**: Verify the service is running and accessible, and that the models have been pulled.
- **Dependency Issues**: Confirm the virtual environment is active and the required packages are installed.
- **Empty Source Content**: Some web pages may not return readable text; try downloading and uploading the content manually.

---

## Project Structure

```
plagiarism/
├── main.py                # Flask application entry point
├── checker.py             # Similarity scoring logic
├── ollama_embeddings.py   # Ollama client for embeddings
├── templates/             # HTML templates (index and results)
├── instance/              # SQLite database location
├── pyproject.toml         # Project metadata and dependencies
└── uv.lock                # Resolved dependency lock file
```

---

## Contributing

1. Fork the repository and create a feature branch.
2. Add tests or run existing checks as appropriate.
3. Submit a pull request describing your changes and reasoning.

---

## License

Specify your licensing terms here (e.g., MIT, Apache 2.0) so others know how they can use and contribute to the project.

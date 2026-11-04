# Health Report Explainer 🩺

An AI-powered web application that explains medical/health reports in simple, easy-to-understand language. Supports 11 Indian languages.

---

## ✨ Features

- **Upload Medical Reports**: JPG, JPEG, PNG, PDF, DOCX (up to 10 MB; legacy DOC files are not supported)
- **AI-Powered Analysis**: Uses Google Gemini to analyze and explain reports
- **Multi-Language Support**: English, Telugu, Tamil, Hindi, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi, Odia
- **Smart OCR**: Reads report images locally using Tesseract OCR
- **Attention Indicators**: Normal / Attention / Discuss with Doctor / Urgent Attention
- **Doctor Discussion Guide**: AI-generated questions to ask your doctor
- **Privacy First**: Reports are processed and deleted, not stored permanently
- **Sample Report**: Try the app with a fictional sample report

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | HTML, CSS, JavaScript |
| **Backend** | Python, FastAPI |
| **AI** | Google Gemini API via the `google-genai` SDK; primary and fallback models are configured in `.env` |
| **PDF Processing** | PyMuPDF for reference indexing; pypdf for uploaded reports |
| **DOCX Processing** | python-docx |
| **Image/OCR** | Tesseract OCR via pytesseract + Pillow; scanned PDF fallback uses Gemini Vision |
| **Medical reference RAG** | Sentence Transformers (`intfloat/multilingual-e5-base`) + persistent ChromaDB |

---

## 📦 Installation

### Prerequisites

- **Python 3.10+** installed ([Download Python](https://www.python.org/downloads/))
- **Google Gemini API Key** ([Get API Key](https://aistudio.google.com/app/apikey))

### Step 1: Clone/Download the project

```bash
cd "Health Report Explainer"
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

### Step 3: Activate the virtual environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Install Tesseract OCR

Image OCR uses the local Tesseract executable; installing the Python requirements alone does not install it.

**Windows:**

1. Install Tesseract OCR using the [Windows installer](https://github.com/UB-Mannheim/tesseract/wiki).
2. During installation, enable the option to add Tesseract to `PATH`. If it was not added automatically, add the installation folder (usually `C:\Program Files\Tesseract-OCR`) to your user or system `PATH`.
3. Open a new terminal and verify the installation:

   ```powershell
   tesseract --version
   ```

**macOS:** `brew install tesseract`

**Debian/Ubuntu:** `sudo apt install tesseract-ocr`

Restart the terminal or backend after changing `PATH`.

### Step 6: Create the `.env` file

```bash
copy .env.example .env
```

### Step 7: Add your Gemini API Key

Open `.env` and replace `your_gemini_api_key_here` with your actual Gemini API key:

```
GEMINI_API_KEY=AIzaSy...your-key-here
```

The first non-empty knowledge-base build or RAG-enabled analysis downloads the pretrained E5 embedding weights if they are not already cached. No model is trained locally. Test-specific explanations are generated only when each test has retrieved evidence and Gemini returns source IDs validated for that test; without evidence, the application reports the limitation and withholds that interpretation.

## 📚 Medical Reference Knowledge Base

Place your own **PDF** or **TXT** references under the matching folder in `data/medical_knowledge/raw/`:

```
data/medical_knowledge/raw/
├── CBC/
├── Diabetes/
├── LFT/
├── KFT/
├── Lipid_Profile/
└── Thyroid/
```

Subfolders are scanned recursively. Index only trusted reference materials; uploaded patient reports are never added to this collection. Source reference documents remain excluded from Git. The generated `data/chroma_db/` index is intended to be included in Git for Docker deployments; it contains extracted reference passages as well as vectors, so only publish it when you have permission to redistribute that content.

Build or incrementally update the persistent collection:

```bash
python scripts/build_knowledge_base.py
```

Test semantic retrieval independently (this does not call Gemini):

```bash
python scripts/test_retrieval.py
```

Chunking and retrieval can be configured with `RAG_CHUNK_SIZE` (default `1000`), `RAG_CHUNK_OVERLAP` (`150`), `RAG_TOP_K` (`4`), `RAG_RELEVANCE_THRESHOLD` (`0.25`), and `RAG_CHROMA_PATH` (`data/chroma_db`). Retrieval status distinguishes an empty knowledge base, no detected tests, insufficient evidence, and successful per-test evidence validation. If evidence is unavailable, unsupported test interpretations are withheld.

## 🚀 Deploying to Render with Docker

The Docker image uses Python 3.11, installs Tesseract with English OCR data, and pre-downloads `intfloat/multilingual-e5-base`. The existing Chroma collection is copied into the image; the image build and application startup both fail if the collection is empty, missing any of the six categories, or cannot retrieve a passage for a category. The database is not rebuilt on each restart.

Before deploying, include the existing files under `data/chroma_db/` in your Git commit. They are no longer ignored by Git or Docker. The raw reference documents remain ignored. Review the database's source and redistribution rights before publishing it.

Create a Render **Web Service** from this repository and select **Docker**. Configure:

| Variable | Required | Default / purpose |
|---|---:|---|
| `GEMINI_API_KEY` | Yes for AI explanations | Store the key as a Render secret; never commit it. |
| `GEMINI_MODEL` | No | `gemini-3.8-flash` |
| `GEMINI_FALLBACK_MODEL` | No | `gemini-3.5-flash-lite` |
| `RAG_CHROMA_PATH` | No | `data/chroma_db` |
| `RAG_CHUNK_SIZE` | No | `1000` |
| `RAG_CHUNK_OVERLAP` | No | `150` |
| `RAG_TOP_K` | No | `4` |
| `RAG_RELEVANCE_THRESHOLD` | No | `0.25` |

Render supplies `PORT`; the container binds Uvicorn to `0.0.0.0` on that port. After deployment, verify `/api/health` and the Render startup log message confirming the knowledge-base chunk and category counts.

The cached E5 model is about 1.13 GB on disk and requires additional runtime memory for Python, PyTorch, ChromaDB, and request processing. Render's Free plan has 512 MB RAM and cannot reliably run this model; use at least the 2 GB `1c-2g` compute plan. The Docker image includes the pre-downloaded model so startup does not depend on downloading it from the Hugging Face Hub.

The knowledge index is baked into the image and is therefore restored on each deploy. Runtime filesystem changes are not durable on Render's Free plan; update the index locally and redeploy rather than modifying it at runtime.

---

## 🚀 Running the Application

### Start the backend server:

```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Open in browser:

```
http://localhost:8000
```

The frontend is served by FastAPI automatically. No separate frontend server is needed!

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/languages` | List supported languages |
| `GET` | `/api/sample-report` | Get sample fictional report |
| `POST` | `/api/upload` | Upload & extract text from report |
| `POST` | `/api/analyze` | Analyze extracted text with AI |
| `POST` | `/api/translate` | Re-generate analysis in different language |

---

## 🔄 How the Pipeline Works

```
Upload → Validate → Extract Text → AI Analysis → Structured JSON → Render Results
```

1. **Upload**: File is validated (type, size) and saved temporarily
2. **Extract**: Text is extracted based on file type:
   - **PDF**: Existing PDF extraction; falls back to Gemini Vision for scanned PDFs. Reference PDFs are extracted with PyMuPDF during knowledge-base indexing.
   - **DOCX**: python-docx paragraph and table extraction
   - **JPG, JPEG, PNG**: Local Tesseract OCR reads the image; the image is not sent to Gemini for text extraction
3. **Analyze**: Extracted text + language → test-scoped retrieval → Gemini JSON explanations with validated source IDs
4. **Display**: Structured JSON response rendered as cards in the dashboard
5. **Translate**: Language switch re-analyzes from extracted text for best quality
6. **Cleanup**: Temporary files are deleted after processing

---

## 🌐 Language Switching

- Select language before analysis, or switch languages on the results page
- Language switching re-generates the analysis from the original extracted text
- Medical values, units, and reference ranges stay unchanged
- Only explanatory text is generated in the selected language

---

## 🔒 Security & Privacy

- API keys are stored only on the backend (`.env` file)
- Uploaded files are validated and sanitized
- Temporary files are deleted after processing
- No permanent storage of medical reports
- Patient information is not displayed unnecessarily
- No raw stack traces exposed to users

---

## ⚕️ Medical Disclaimer

This tool provides AI-generated explanations for **educational and informational purposes only**. It is **not** a medical diagnosis or substitute for professional medical advice. Always consult a qualified healthcare professional for diagnosis and treatment.

---

## 📁 Project Structure

```
Health Report Explainer/
├── backend/
│   ├── rag/                   # E5 embeddings, ingestion, ChromaDB, retrieval and RAG prompt context
│   ├── __init__.py
│   ├── main.py              # FastAPI app, routes, static serving
│   ├── config.py             # Settings from environment variables
│   ├── models/
│   │   ├── __init__.py
│   │   └── report.py         # Pydantic data models
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── upload.py          # File upload endpoint
│   │   └── analysis.py        # Analysis & translation endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ai_service.py      # Gemini AI integration
│   │   └── document_service.py # PDF/DOCX/Image text extraction
│   └── utils/
│       ├── __init__.py
│       └── validators.py      # File validation utilities
├── data/
│   ├── medical_knowledge/
│   │   ├── raw/               # Add trusted PDF/TXT references by category
│   │   └── processed/
│   └── chroma_db/             # Persistent ChromaDB files (generated)
├── scripts/
│   ├── build_knowledge_base.py
│   └── test_retrieval.py
├── frontend/
│   ├── index.html             # Single-page application
│   ├── css/
│   │   └── styles.css         # Complete design system
│   ├── js/
│   │   └── app.js             # Frontend logic
│   └── assets/
├── .env.example               # Environment variables template
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── TEST_REPORT.md             # Testing documentation
```

---

## 🔮 Known Limitations

- OCR accuracy depends on image quality
- AI analysis quality depends on report clarity
- Very large or complex multi-page reports may take longer
- AI may occasionally miss or misinterpret handwritten text
- Language quality varies by language (major languages work better)
- No user authentication (designed as a single-user tool)

---

## 🚀 Future Improvements

- User accounts and report history
- Comparison between multiple reports over time
- Export analysis as PDF
- More languages
- Custom AI model fine-tuning for medical reports
- Batch processing of multiple reports
- Voice explanation (text-to-speech)
- WhatsApp/Telegram bot integration

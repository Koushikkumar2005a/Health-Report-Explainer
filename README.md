# Health Report Explainer 🩺

An AI-powered web application that explains medical/health reports in simple, easy-to-understand language. Supports 11 Indian languages.

---

## ✨ Features

- **Upload Medical Reports**: JPG, JPEG, PNG, PDF, DOC, DOCX (up to 10 MB)
- **AI-Powered Analysis**: Uses Google Gemini to analyze and explain reports
- **Multi-Language Support**: English, Telugu, Tamil, Hindi, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi, Odia
- **Smart OCR**: Reads scanned reports and images using Gemini Vision
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
| **PDF Processing** | PyMuPDF (fitz) |
| **DOCX Processing** | python-docx |
| **Image/OCR** | Gemini Vision API + Pillow |

---

## 📦 Installation

### Prerequisites

- **Python 3.9+** installed ([Download Python](https://www.python.org/downloads/))
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

### Step 5: Create the `.env` file

```bash
copy .env.example .env
```

### Step 6: Add your Gemini API Key

Open `.env` and replace `your_gemini_api_key_here` with your actual Gemini API key:

```
GEMINI_API_KEY=AIzaSy...your-key-here
```

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
   - **PDF**: PyMuPDF text extraction; falls back to Gemini Vision for scanned PDFs
   - **DOCX**: python-docx paragraph and table extraction
   - **Images**: Gemini Vision API reads the image
3. **Analyze**: Extracted text + language → Gemini API with medical analysis prompt
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

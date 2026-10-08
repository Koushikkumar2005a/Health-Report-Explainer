# Health Report Explainer — Test Report 🧪

## Testing Checklist

This document outlines the test cases for the Health Report Explainer application.

---

## 1. File Upload Tests

| # | Test Case | Steps | Expected Result | Status |
|---|-----------|-------|-----------------|--------|
| 1 | JPG Upload | Upload a .jpg medical report image | Text extracted, analysis shown | ⬜ |
| 2 | JPEG Upload | Upload a .jpeg medical report image | Text extracted, analysis shown | ⬜ |
| 3 | PNG Upload | Upload a .png medical report image | Text extracted, analysis shown | ⬜ |
| 4 | Text PDF | Upload a text-based PDF report | Text extracted directly, analysis shown | ⬜ |
| 5 | Scanned PDF | Upload a scanned/image PDF | OCR via Gemini Vision, analysis shown | ⬜ |
| 6 | DOCX | Upload a .docx Word document | Text extracted, analysis shown | ⬜ |

## 2. File Validation Tests

| # | Test Case | Steps | Expected Result | Status |
|---|-----------|-------|-----------------|--------|
| 7 | Invalid File Type | Upload a .txt or .exe file | "Unsupported file type" error shown | ⬜ |
| 8 | File > 10 MB | Upload a file larger than 10 MB | "File too large" error shown | ⬜ |
| 9 | Empty File | Upload an empty file | "File is empty" error shown | ⬜ |

## 3. Language Tests

| # | Test Case | Steps | Expected Result | Status |
|---|-----------|-------|-----------------|--------|
| 10 | English Explanation | Analyze with English selected | English explanation shown | ⬜ |
| 11 | Telugu Explanation | Analyze with Telugu selected | Telugu explanation shown | ⬜ |
| 12 | Tamil Explanation | Analyze with Tamil selected | Tamil explanation shown | ⬜ |
| 13 | Hindi Explanation | Analyze with Hindi selected | Hindi explanation shown | ⬜ |
| 14 | Language Switching | Change language on results page | New language explanation generated without re-upload | ⬜ |

## 4. AI Analysis Tests

| # | Test Case | Steps | Expected Result | Status |
|---|-----------|-------|-----------------|--------|
| 15 | AI API Failure | Use invalid API key | "Unable to analyze" error shown | ⬜ |
| 16 | OCR Failure | Upload unreadable image | "Could not read" error shown | ⬜ |
| 17 | Missing Values | Upload partial report | "Not available in the report" shown for missing info | ⬜ |
| 18 | Abnormal Values | Upload report with out-of-range values | Abnormal findings highlighted with attention level | ⬜ |
| 19 | Multiple Tests | Upload report with many test results | All tests listed in results table | ⬜ |

## 5. UI/UX Tests

| # | Test Case | Steps | Expected Result | Status |
|---|-----------|-------|-----------------|--------|
| 20 | Mobile UI | Open on mobile viewport (375px) | Responsive layout, readable text, functional buttons | ⬜ |
| 21 | Navigation | Click all nav links | Pages/sections navigate correctly | ⬜ |
| 22 | Drag & Drop | Drag file to upload zone | File accepted, info shown | ⬜ |
| 23 | Consent Required | Try to analyze without consent | Analyze button disabled | ⬜ |
| 24 | Sample Report | Click "Try Sample Report" | Sample loaded, can analyze | ⬜ |
| 25 | Processing Screen | Start analysis | Animated step progression shown | ⬜ |
| 26 | Remove File | Click remove after selecting file | File cleared, upload zone reset | ⬜ |
| 27 | New Report | Click "Analyze Another Report" | Upload page shown with reset state | ⬜ |

## 6. Security Tests

| # | Test Case | Steps | Expected Result | Status |
|---|-----------|-------|-----------------|--------|
| 28 | API Key Hidden | Inspect frontend JS | No API key visible in frontend code | ⬜ |
| 29 | File Cleanup | Upload and analyze, check temp_uploads/ | Temporary file deleted after processing | ⬜ |
| 30 | Filename Sanitization | Upload file with special characters in name | Filename sanitized, no path traversal | ⬜ |

---

## How to Test

### Quick Test (Sample Report)

1. Start the backend: `python -m uvicorn backend.main:app --reload --port 8000`
2. Open `http://localhost:8000`
3. Click "Upload Report" or "Explain My Report"
4. Click "Try Sample Report"
5. Check the consent checkbox
6. Click "Analyze Report"
7. Wait for analysis to complete
8. Review the results dashboard
9. Switch language to Telugu
10. Verify Telugu explanation is generated

### Full Test (Real Report)

1. Prepare a medical report (blood test, CBC, etc.)
2. Take a photo or use the PDF
3. Upload it through the application
4. Verify text extraction
5. Verify AI analysis
6. Test multiple languages
7. Test with different file formats

---

## Test Status Legend

- ⬜ Not tested
- ✅ Passed
- ❌ Failed
- ⚠️ Partially passed

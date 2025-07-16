# 📚 NoteMate AI

A modern web application that helps students **generate study notes directly from their course syllabi**.  
Built with a **React + Tailwind CSS frontend** and a **Python FastAPI backend**, it combines modern UI with AI-powered note generation (planned).

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)

---

## ✨ Features

- 📄 **Syllabus Upload (placeholder):** Upload a syllabus PDF (currently loads sample data).
- 🧩 **Structured Syllabus Display:** Expandable, organized view of syllabus units and topics.
- ✅ **Topic Selection:** Select specific topics/units to generate notes.
- 📝 **Notes Generation:** Backend process to create study notes (implementation planned).

---

## 🛠 Technologies Used

### Frontend
- **React**
- **Tailwind CSS**

### Backend
- **Python 3.11**, **FastAPI**, **Uvicorn**
- **PyPDF2**, **python-docx**, **python-magic**
- **SQLAlchemy**, **pydantic**, **python-dotenv**
- **loguru**, **python-json-logger**
- **python-jose[cryptography]** for JWT authentication
- **transformers**, **torch**, **langchain**, **sentence-transformers**
- **chromadb**, **pandas**, **numpy**, **scipy**
- **requests**, **httpx**

---

## ⚙️ Installation & Setup

> **Recommended:** Use a Python virtual environment.

### 1️⃣ Clone the repository
```bash
git clone <your-repository-url>
cd <project-folder>
```
# Remove old venv (optional)
```bash
rm -rf venv
```
# Create venv (adjust Python path if needed)
```bash
python3.11 -m venv venv
```
# Activate
```bash
source venv/bin/activate
```
3️⃣ Install backend dependencies
```bash
pip install fastapi "uvicorn[standard]" python-multipart \
PyPDF2==3.0.1 python-docx==1.1.0 python-magic==0.4.27 \
python-dotenv==1.0.0 sqlalchemy==2.0.23 loguru==0.7.2 \
python-json-logger==2.0.7 pytest==7.4.3 pytest-asyncio==0.21.1 \
"python-jose[cryptography]==3.3.0" requests httpx chromadb \
sentence-transformers langchain langchain-community langchain-chroma \
pandas numpy scipy pydantic pydantic-settings transformers torch
```
▶️ Running the Application
🚀 Start the backend
```bash
uvicorn main:app --reload
```
Default: http://127.0.0.1:8000

💻 Start the frontend
From your frontend folder:
```bash
npm install
npm run dev
```
Default: http://localhost:5173

🧪 Usage (prototype)
Open the frontend in your browser.

Upload a syllabus PDF (currently loads static sample data).

Select desired units/topics.

Click Generate Notes (backend AI note generation to be added).

📦 Project Structure (conceptual)
```bash
/src
  ├─ App.js                # Main React app
  └─ ChapterSelection.js   # Syllabus display & selection
/backend
  └─ main.py               # FastAPI entry point
/venv                      # Python virtual environment
```
🚀 Future Enhancements
✅ Robust PDF parsing
✅ AI-powered, personalized notes
✅ "Best note" comparison from online resources
✅ User authentication & profiles
✅ Database storage (SQLAlchemy)
✅ Mobile-friendly, improved UI/UX

🤝 Contributing
Contributions, issues, and feature requests are welcome!
Feel free to open an issue or submit a pull request.

📄 License
Distributed under the MIT License.
See LICENSE for more information.

⭐️ Show your support
If you find this project useful, please ⭐️ the repo!

Made with ❤️ using React, FastAPI & AI


---

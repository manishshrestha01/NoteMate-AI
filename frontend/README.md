# Syllabus Notes Generator

This project is a web application designed to help students generate study notes from their course syllabi. It features a React-based frontend for displaying syllabus units and selecting topics, coupled with a Python backend for processing PDF files and potentially generating notes.

## Features

* **Syllabus Upload (Placeholder):** Allows users to "upload" a syllabus PDF. (Note: Actual PDF parsing logic needs to be implemented in the backend.)
* **Structured Syllabus Display:** Presents course units and their detailed content in an organized, expandable format on the frontend.
* **Topic Selection:** Enables users to select specific units/topics for which they want to generate study notes.
* **Notes Generation:** Triggers a process to generate notes based on selected topics. (Backend implementation required for actual note generation.)

## Technologies Used

This project leverages a combination of modern web and AI technologies:

### Frontend
* **React:** A JavaScript library for building user interfaces.
* **Tailwind CSS:** A utility-first CSS framework for rapid UI development.

### Backend (Based on provided dependencies)
* **Python:** The primary language for the backend.
* **FastAPI:** A modern, fast (high-performance) web framework for building APIs with Python 3.7+.
* **Uvicorn:** An ASGI server for running FastAPI applications.
* **PyPDF2:** A Python library for working with PDF files (likely for text extraction).
* **python-docx:** A library for creating and updating Microsoft Word files.
* **python-magic:** A Python wrapper for `libmagic` to determine file types.
* **python-dotenv:** For loading environment variables.
* **SQLAlchemy:** An SQL toolkit and Object Relational Mapper (ORM) for database interactions.
* **loguru:** For easy and flexible logging.
* **python-json-logger:** For JSON logging.
* **python-jose[cryptography]:** For JWT (JSON Web Token) authentication.
* **requests & httpx:** Libraries for making HTTP requests.
* **chromadb:** An open-source embedding database for building AI applications.
* **sentence-transformers:** For generating embeddings (vector representations of text).
* **langchain & langchain-community:** Frameworks for developing applications powered by language models.
* **langchain-google-genai:** Integration with Google's Generative AI models.
* **pandas, numpy, scipy:** Fundamental libraries for data manipulation and scientific computing.
* **pydantic & pydantic-settings:** For data validation and settings management.
* **transformers & torch:** Hugging Face's Transformers library and PyTorch, commonly used for working with large language models.

## Setup and Installation

Follow these steps to set up the project locally:

### 1. Clone the Repository
git clone <your-repository-url>

### 2. Create and Activate a Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.

# Deactivate any existing virtual environment (if applicable)
deactivate 

# Remove any old 'venv' directory (optional, for a clean start)
rm -rf venv

# Create a new virtual environment
/opt/homebrew/bin/python3.11 -m venv venv 

# Activate the virtual environment
source venv/bin/activate

### 3. Install Backend Dependencies
Install all the required Python packages for the backend.

pip install fastapi "uvicorn[standard]" python-multipart PyPDF2==3.0.1 python-docx==1.1.0 python-magic==0.4.27 python-dotenv==1.0.0 sqlalchemy==2.0.23 loguru==0.7.2 python-json-logger==2.0.7 pytest==7.4.3 pytest-asyncio==0.21.1 "python-jose[cryptography]==3.3.0" requests httpx chromadb sentence-transformers langchain langchain-community langchain-chroma pandas numpy scipy pydantic pydantic-settings transformers torch


### Running the Application
1. Start the Backend Server
From your project's root directory (where main.py is located, assuming your FastAPI app is in main.py):

uvicorn main:app --reload

This will start the FastAPI server, typically on http://127.0.0.1:8000.

2. Start the Frontend Development Server
From your frontend directory (e.g., syllabus-app if you used create-react-app there):

npm run dev

This will start the React development server, typically on http://localhost:5173.


Usage
Access the frontend application in your web browser (e.g., http://localhost:3000).

Use the file upload section to "upload" a syllabus PDF. (Currently, this loads a sample hardcoded data; integrate your backend processing here to dynamically load from actual PDFs.)

Select the desired units or topics for which you want to generate notes.

Click the "Generate Notes" button. The backend would then process this request to provide the notes.

Project Structure (Conceptual)
src/App.js: Main React component managing application state and routing.

src/ChapterSelection.js: React component for displaying syllabus units and handling user selections.

backend/main.py: (Assumed) FastAPI application entry point for API routes, PDF processing, and note generation logic.

venv/: Python virtual environment.

Future Enhancements (Ideas)
Robust PDF Parsing: Implement comprehensive backend logic to accurately extract structured content from various PDF syllabus formats.

Advanced Note Generation: Utilize the integrated LLMs (transformers, langchain-google-genai) to generate more detailed, coherent, and personalized notes based on extracted syllabus topics.

"Best Note" Comparison: (As discussed, a significant undertaking) Develop a system to search online resources, compare notes for quality and relevance, and present the most suitable ones. This would involve web scraping/APIs and advanced NLP/AI.

User Authentication: Add user login/registration functionalities.

Database Integration: Store user data, uploaded syllabi, and generated notes in a database (e.g., using SQLAlchemy).

Improved UI/UX: Enhance the user interface for a more intuitive and visually appealing experience.









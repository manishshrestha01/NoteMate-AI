// src/App.jsx
import React, { useState } from 'react';
import UploadForm from "./components/UploadForm";
import Navbar from "./components/Navbar";
import Display from "./components/Display";
import ChapterSelection from "./components/ChapterSelection"; // Import the new component

function App() {
  // --- Core States ---
  const [currentView, setCurrentView] = useState('upload'); // 'upload', 'selectChapters', 'displayNotes'
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadedFileName, setUploadedFileName] = useState('');
  const [currentSyllabusId, setCurrentSyllabusId] = useState(null); // <--- NEW: State to store the syllabus_id

  // --- UI Feedback States ---
  const [uploadLoading, setUploadLoading] = useState(false); // For PDF upload process
  const [notesLoading, setNotesLoading] = useState(false); // For notes generation process
  const [uploadError, setUploadError] = useState('');
  const [dragActive, setDragActive] = useState(false);

  // --- Data States ---
  const [extractedChapters, setExtractedChapters] = useState([]); // Stores units_data from syllabus processing
  const [generatedStudyNotes, setGeneratedStudyNotes] = useState([]); // Stores notes from backend for Display

  // --- Configuration ---
  const supportedFormats = {
    supported_formats: ['.pdf'],
    descriptions: { pdf: 'Portable Document Format' },
    ocr_available: true,
  };

  // --- Helper Functions ---
  const isValidFileType = (file) => file && file.type === 'application/pdf';

  const resetAllStates = () => {
    setCurrentView('upload');
    setSelectedFile(null);
    setUploadedFileName('');
    setCurrentSyllabusId(null); // <--- NEW: Reset syllabus ID
    setUploadLoading(false);
    setNotesLoading(false);
    setUploadError('');
    setDragActive(false);
    setExtractedChapters([]);
    setGeneratedStudyNotes([]);
  };

  // Handles file selection (from input or drop)
  const handleFileSelect = (file) => {
    setUploadError(''); // Clear previous errors
    if (file) {
      if (isValidFileType(file)) {
        setSelectedFile(file);
        setUploadedFileName(file.name);
        setUploadError('');
        // Automatically trigger upload on select/drop
        handleUpload(file);
      } else {
        setUploadError('Invalid file type. Only PDF files are allowed.');
        setSelectedFile(null);
        setUploadedFileName('');
      }
    }
  };

  // Handles the actual PDF upload to FastAPI
  const handleUpload = async (fileToProcess) => {
    if (!fileToProcess) {
      setUploadError('No file selected to upload.');
      return;
    }
    if (!isValidFileType(fileToProcess)) {
      setUploadError('Invalid file type. Only PDF files are allowed.');
      return;
    }

    setUploadLoading(true);
    setUploadError(''); // Clear previous errors

    const formData = new FormData();
    formData.append('file', fileToProcess);

    try {
      const response = await fetch('http://127.0.0.1:8000/process-syllabus/', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        // Assuming backend returns units_data and syllabus_id
        setExtractedChapters(data.units_data || []);
        setCurrentSyllabusId(data.syllabus_id); // <--- NEW: Store the syllabus_id
        setCurrentView('selectChapters'); // Move to chapter selection view
      } else {
        const errorData = await response.json();
        setUploadError(errorData.detail || 'Unknown error occurred during processing.');
      }
    } catch (error) {
      setUploadError(`An error occurred: ${error.message}`);
      console.error('Fetch Error:', error);
    } finally {
      setUploadLoading(false);
    }
  };

  // Handles the generation of notes based on selected chapters
  const handleGenerateNotes = async (selectedTopics) => {
    if (selectedTopics.length === 0) {
      setUploadError('Please select at least one chapter to generate notes.');
      return;
    }

    // You can choose the source type here. For now, we'll default to 'syllabus'.
    // If you add a UI element to switch between 'syllabus' and 'internet',
    // you would pass that value here.
    const sourceType = 'syllabus'; // Defaulting to syllabus for now

    // IMPORTANT: Construct the request body as a JSON object
    const requestBody = {
      topics: selectedTopics,
      source_type: sourceType,
      // syllabus_id is REQUIRED if sourceType is 'syllabus'
      ...(sourceType === 'syllabus' && { syllabus_id: currentSyllabusId }),
      // You could add an 'internet_search_query' field if source_type is 'internet' and you want to customize the search
    };

    // Basic validation for syllabus_id if source type is syllabus
    if (sourceType === 'syllabus' && !currentSyllabusId) {
        setUploadError('Syllabus ID is missing. Please upload a syllabus first.');
        return;
    }


    setNotesLoading(true);
    setUploadError(''); // Clear previous errors
    setGeneratedStudyNotes([]); // Clear previous notes

    try {
      const response = await fetch('http://127.0.0.1:8000/generate-notes/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody), // <--- CORRECTED: Send the full requestBody object
      });

      if (response.ok) {
        const data = await response.json();
        setGeneratedStudyNotes(data.notes || []); // Assuming backend returns { "notes": [...] }
        setCurrentView('displayNotes'); // Move to notes display view
      } else {
        const errorData = await response.json();
        console.error('Backend Error Response:', errorData); // Log error for debugging
        setUploadError(errorData.detail || 'Unknown error occurred during note generation.');
      }
    } catch (error) {
      setUploadError(`An error occurred: ${error.message}`);
      console.error('Fetch Error:', error);
    } finally {
      setNotesLoading(false);
    }
  };


  // --- Drag and Drop Handlers (passed to UploadForm) ---
  const onDragEnter = (e) => { e.preventDefault(); e.stopPropagation(); setDragActive(true); };
  const onDragLeave = (e) => { e.preventDefault(); e.stopPropagation(); setDragActive(false); };
  const onDragOver = (e) => { e.preventDefault(); e.stopPropagation(); setDragActive(true); };
  const onDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  // --- Main Render Logic ---
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <Navbar />
      <main className="container mx-auto px-4 py-8 flex-grow">
        <div className="max-w-4xl mx-auto">
          {currentView === 'upload' && (
            <UploadForm
              onFileUpload={(e) => handleFileSelect(e.target.files[0])}
              loading={uploadLoading}
              error={uploadError}
              dragActive={dragActive}
              onDragEnter={onDragEnter}
              onDragLeave={onDragLeave}
              onDragOver={onDragOver}
              onDrop={onDrop}
              supportedFormats={supportedFormats}
            />
          )}

          {currentView === 'selectChapters' && (
            <ChapterSelection
              unitsData={extractedChapters}
              onGenerateNotes={handleGenerateNotes}
              onReset={resetAllStates}
              fileName={uploadedFileName}
              isLoadingNotes={notesLoading}
            />
          )}

          {currentView === 'displayNotes' && (
            <Display
              notes={generatedStudyNotes}
              onReset={resetAllStates}
              fileName={uploadedFileName}
            />
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
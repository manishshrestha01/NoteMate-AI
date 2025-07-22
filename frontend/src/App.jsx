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
  const [currentSyllabusId, setCurrentSyllabusId] = useState(null);

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
    setCurrentSyllabusId(null);
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
        setExtractedChapters(data.units_data || []);
        setCurrentSyllabusId(data.syllabus_id);
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

    const sourceType = 'syllabus'; // Defaulting to syllabus for now

    const requestBody = {
      topics: selectedTopics,
      source_type: sourceType,
      ...(sourceType === 'syllabus' && { syllabus_id: currentSyllabusId }),
    };

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
        body: JSON.stringify(requestBody),
      });

      if (response.ok) {
        const data = await response.json();
        setGeneratedStudyNotes(data.notes || []);
        setCurrentView('displayNotes'); // Move to notes display view
      } else {
        const errorData = await response.json();
        console.error('Backend Error Response:', errorData);
        setUploadError(errorData.detail || 'Unknown error occurred during note generation.');
      }
    } catch (error) {
      setUploadError(`An error occurred: ${error.message}`);
      console.error('Fetch Error:', error);
    } finally {
      setNotesLoading(false);
    }
  };

  // --- NEW: Function to go back from 'displayNotes' to 'selectChapters' ---
  const handleBackToSyllabusSelection = () => {
    setCurrentView('selectChapters');
    // Optionally clear generated notes when going back, if you want
    setGeneratedStudyNotes([]);
  };

  // Handles the generation of AI notes based on selected chapters
  const handleGenerateAiNotes = async (selectedTopics) => {
    if (selectedTopics.length === 0) {
      setUploadError('Please select at least one chapter to generate notes.');
      return;
    }
    if (!currentSyllabusId) {
      setUploadError('Syllabus ID is missing. Please upload a syllabus first.');
      return;
    }

    setNotesLoading(true);
    setUploadError('');
    setGeneratedStudyNotes([]);

    const requestBody = {
      topics: selectedTopics,
      source_type: 'syllabus',
      syllabus_id: currentSyllabusId,
    };

    try {
      const response = await fetch('http://127.0.0.1:8000/generate-ai-notes/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });

      if (response.ok) {
        const data = await response.json();
        setGeneratedStudyNotes(data);
        setCurrentView('displayNotes');
      } else {
        const errorData = await response.json();
        setUploadError(errorData.detail || 'Unknown error occurred during AI note generation.');
      }
    } catch (error) {
      setUploadError(`An error occurred: ${error.message}`);
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
              onGenerateAiNotes={handleGenerateAiNotes}
              onReset={resetAllStates} // Use resetAllStates to go back to upload
              fileName={uploadedFileName}
              isLoadingNotes={notesLoading}
            />
          )}

          {currentView === 'displayNotes' && (
            <Display
              notes={generatedStudyNotes}
              onReset={resetAllStates} // This button will take user to 'upload' view
              fileName={uploadedFileName}
              // --- NEW PROP: Pass the back handler to Display component ---
              onBackToSyllabus={handleBackToSyllabusSelection}
            />
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
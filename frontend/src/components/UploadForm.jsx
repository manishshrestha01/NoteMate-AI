import React, { useRef } from 'react';

const UploadForm = ({ onFileUpload, loading, error, dragActive, onDragEnter, onDragLeave, onDragOver, onDrop, supportedFormats }) => {
  const inputRef = useRef(null);

  const handleButtonClick = () => {
    inputRef.current.click();
  };

  return (
    <div className="bg-white p-8 rounded-2xl shadow-xl border border-gray-100"> {/* Matched ChapterSelection container */}
      <h2 className="text-3xl font-extrabold text-gray-900 mb-3"> {/* Matched ChapterSelection heading */}
        Upload Your <span className="text-blue-600">Syllabus</span>
      </h2>
      <p className="text-gray-700 text-lg mb-8 leading-relaxed"> {/* Matched ChapterSelection paragraph */}
        Please upload your course syllabus here to get started. Our AI will analyze it to extract chapters and key topics.
      </p>

      {/* Main drag-and-drop area */}
      <div
        className={`p-10 border-2 rounded-xl transition-all duration-200 flex flex-col items-center justify-center text-center
          ${dragActive ? 'bg-blue-50 border-blue-400' : 'bg-gray-50 border-gray-300'}
        `}
        onDragEnter={onDragEnter}
        onDragLeave={onDragLeave}
        onDragOver={onDragOver}
        onDrop={onDrop}
      >
        <input
          type="file"
          ref={inputRef}
          onChange={onFileUpload}
          accept={supportedFormats.supported_formats.map(format => `.${format}`).join(',')}
          className="hidden"
          disabled={loading}
        />

        {loading ? (
          <div className="flex flex-col items-center justify-center py-8"> {/* Slightly adjusted padding */}
            <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-5"></div> {/* Larger spinner */}
            <p className="text-blue-600 font-semibold text-xl">Processing your syllabus...</p> {/* Larger text */}
            <p className="text-gray-500 text-base mt-2">This may take a moment to extract chapters and sub-units.</p> {/* Larger text */}
          </div>
        ) : (
          <>
            <svg
              className={`mx-auto h-20 w-20 ${dragActive ? 'text-blue-500' : 'text-blue-500'} transition-colors duration-200`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              {/* Corrected SVG path for a cloud upload icon */}
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
            </svg>
            <p className="mt-6 text-gray-700 font-medium text-lg"> {/* Larger text */}
              {dragActive ? "Release to upload your file!" : "Drag & drop your PDF file here"}
            </p>
            <p className="mt-3 text-sm text-gray-500">
              Supported formats: <span className="font-semibold text-gray-600">
                {supportedFormats.supported_formats.map(format => format.toUpperCase()).join(', ')}
              </span>
            </p>
            <p className="mt-4 text-sm text-gray-500">
              or
            </p>
            <button
              onClick={handleButtonClick}
              className="mt-5 px-8 py-4 bg-blue-600 text-white rounded-lg font-semibold text-lg hover:bg-blue-700 transition-colors shadow-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
              disabled={loading}
            >
              Browse Files
            </button>
          </>
        )}

        {error && (
          // Matched ChapterSelection's error message style
          <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-lg flex items-center mt-6 w-full">
            <svg className="w-6 h-6 mr-3 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>
            </svg>
            <p className="font-medium">{error}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default UploadForm;
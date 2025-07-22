import React from 'react';

// IMPORTANT: Add 'onBackToSyllabus' to the destructured props
const Display = ({ notes, onReset, fileName, onBackToSyllabus }) => {
  return (
    <div className="bg-white p-8 rounded-2xl shadow-xl border border-gray-100">
      <h2 className="text-3xl font-extrabold text-gray-900 mb-3">
        Generated Study <span className="text-blue-600">Materials</span>
      </h2>
      <p className="text-gray-700 text-lg mb-4 leading-relaxed">
        AI-curated resources for your selected topics.
      </p>
      {fileName && (
        <p className="text-base text-gray-600 mb-8">
          <span className="font-semibold">Based on:</span> {fileName}
        </p>
      )}

      {notes.length === 0 ? (
        <div className="text-center py-10 text-gray-500">
          <div className="bg-yellow-50 border border-yellow-200 text-yellow-700 p-4 rounded-lg flex items-center justify-center mb-6">
            <svg className="w-6 h-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
            <p className="font-medium">No study materials found for the selected topics.</p>
          </div>
          {/* You might want a "Back to Syllabus" here too if no notes are found,
              or keep "Start Over" as it is, depending on desired flow. */}
          <button
            onClick={onReset}
            className="px-6 py-3 text-gray-700 font-medium hover:bg-gray-50 border border-gray-300 rounded-lg shadow-sm transition duration-150 ease-in-out focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            Start Over
          </button>
        </div>
      ) : (
        <>
          <div className="flex items-center justify-between mb-8 pb-4 border-b border-gray-100">
            <span className="text-sm font-semibold text-gray-700 bg-blue-50 px-4 py-2 rounded-full">
              {notes.length} resources generated
            </span>
          </div>

          <div className="space-y-6" role="list" aria-label="Generated study materials">
            {notes.map((note, index) => (
              <div
                key={index}
                className="bg-gray-50 border border-gray-200 rounded-xl p-6 shadow-sm overflow-hidden hover:shadow-md transition-shadow duration-200"
                role="listitem"
              >
                <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between sm:gap-4">
                  <div className="flex-1">
                    <div className="flex items-center flex-wrap gap-2 mb-3">
                      <span className="text-xs font-semibold text-blue-700 bg-blue-100 px-3 py-1 rounded-full">
                        {note.topic || note.topicTitle}
                      </span>
                      {/* Only show source if it exists */}
                      {note.source && (
                        <span
                          className={`text-xs px-3 py-1 rounded-full font-semibold ${
                            note.source === 'syllabus_excerpt' ? 'bg-indigo-100 text-indigo-700' :
                            note.source === 'web_search' ? 'bg-teal-100 text-teal-700' :
                            note.source === 'educational' ? 'bg-green-100 text-green-700' :
                            note.source === 'github' ? 'bg-purple-100 text-purple-700' :
                            note.source === 'video' ? 'bg-red-100 text-red-700' :
                            note.source === 'general' ? 'bg-yellow-100 text-yellow-700' :
                            note.source === 'error' ? 'bg-red-100 text-red-700' :
                            'bg-gray-200 text-gray-800'
                          }`}
                          aria-label={`Source: ${note.source || "unknown"}`}
                        >
                          {(note.source || "").replace(/_/g, ' ')}
                        </span>
                      )}
                    </div>
                    {/* For AI notes, show the note content, for web notes show title/snippet/url */}
                    {(note.download_url || note.note_id) ? (
                      <>
                        <h3 className="font-bold text-gray-900 text-xl mb-2">Click the Download button to download your note.</h3>
                        <a
                          href={`http://127.0.0.1:8000${note.download_url || `/download-note/${note.note_id}`}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-2 text-blue-600 hover:text-blue-800 text-base font-semibold focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded-md transition duration-150 ease-in-out mb-2 px-4 py-2 bg-blue-100 border border-blue-300"
                        >
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                          </svg>
                          Download for {note.topic || note.topicTitle}
                        </a>
                      </>
                    ) : (
                      <>
                        <h3 className="font-bold text-gray-900 text-xl mb-2">{note.title}</h3>
                        <p className="text-gray-700 text-sm mb-4 leading-relaxed">{note.snippet}</p>
                        {note.url && (
                          <a
                            href={note.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-2 text-blue-600 hover:text-blue-800 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded-md transition duration-150 ease-in-out"
                            aria-label={`Access resource: ${note.title} (opens in new tab)`}
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                            </svg>
                            Access Resource
                            <span className="sr-only">(opens in new tab)</span>
                          </a>
                        )}
                      </>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* This is the div where you'll add the back button */}
          <div className="mt-10 flex justify-between border-t border-gray-200 pt-8">
            {/* NEW: Back to Syllabus Selection Button */}
            <button
              onClick={onBackToSyllabus}
              className="px-6 py-3 border border-gray-300 rounded-lg text-gray-700 font-medium hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition duration-150 ease-in-out shadow-sm"
            >
              ← Back to Syllabus Selection
            </button>

            {/* Existing Upload New Syllabus button */}
            <button
              onClick={onReset}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition duration-150 ease-in-out shadow-md"
            >
              Upload New Syllabus
            </button>
          </div>
        </>
      )}
    </div>
  );
};

export default Display;
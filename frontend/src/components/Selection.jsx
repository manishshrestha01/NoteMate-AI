import React from 'react';

const Selection = ({ 
  topics, 
  selectedTopics, 
  toggleTopicSelection, 
  onGenerateNotes, 
  onReset, 
  searchLoading,
  fileName 
}) => {
  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden">
      <div className="px-8 py-6 border-b border-gray-100">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-gray-800">Select Topics for Note Generation</h2>
            <p className="text-sm text-gray-500 mt-1">Choose which chapters you'd like to find study materials for</p>
            {fileName && (
              <p className="text-xs text-blue-600 mt-1">From: {fileName}</p>
            )}
          </div>
          <span className="text-sm text-gray-500 bg-gray-100 px-3 py-1 rounded-full">
            {topics.length} topics found
          </span>
        </div>
      </div>
      
      <div className="p-8">
        <div className="space-y-4 mb-8" role="group" aria-label="Select topics for note generation">
          {topics.map((topic) => (
            <div key={topic.id} className="border border-gray-200 rounded-xl p-6 hover:shadow-sm transition-shadow">
              <label className="flex items-start gap-4 cursor-pointer">
                <input
                  type="checkbox"
                  checked={selectedTopics.includes(topic.id)}
                  onChange={() => toggleTopicSelection(topic.id)}
                  className="mt-1 w-5 h-5 text-blue-600 border-gray-300 rounded focus:ring-blue-500 focus:ring-2"
                  aria-describedby={`topic-${topic.id}-description`}
                />
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <h3 className="font-semibold text-gray-800" id={`topic-${topic.id}-title`}>
                      {topic.title}
                    </h3>
                    {topic.hours && (
                      <span className="text-xs text-blue-600 bg-blue-100 px-2 py-1 rounded">
                        {topic.hours} hours
                      </span>
                    )}
                  </div>
                  {topic.contents.length > 0 && (
                    <ul className="space-y-1" id={`topic-${topic.id}-description`}>
                      {topic.contents.map((content, idx) => (
                        <li key={idx} className="text-sm text-gray-600 flex items-start gap-2">
                          <span className="w-1.5 h-1.5 bg-gray-400 rounded-full mt-2 flex-shrink-0" aria-hidden="true"></span>
                          {content}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </label>
            </div>
          ))}
        </div>

        <div className="flex justify-center gap-4">
          <button
            onClick={onReset}
            className="px-6 py-2 text-gray-600 hover:text-gray-800 border border-gray-300 hover:border-gray-400 rounded-lg font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-gray-300"
            aria-label="Upload a new syllabus file"
          >
            Upload New File
          </button>
          <button
            onClick={onGenerateNotes}
            disabled={selectedTopics.length === 0 || searchLoading}
            className="px-8 py-3 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors flex items-center gap-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            aria-label={`Generate notes for ${selectedTopics.length} selected topic${selectedTopics.length !== 1 ? 's' : ''}`}
          >
            {searchLoading ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" aria-hidden="true"></div>
                <span>Generating Notes...</span>
                <span className="sr-only">Please wait while we generate your notes</span>
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                </svg>
                Generate Notes ({selectedTopics.length} selected)
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default Selection;
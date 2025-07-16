import React, { useState, useEffect } from 'react';

function ChapterSelection({ unitsData, onGenerateNotes, onReset, fileName, isLoadingNotes }) {
  // State to store which topics (unit titles or sub-unit titles) are selected for notes generation
  const [selectedTopics, setSelectedTopics] = useState([]);
  // State to manage which unit's content is currently open/expanded
  const [openUnitId, setOpenUnitId] = useState(null); // Stores the ID or index of the currently open unit

  useEffect(() => {
    // If unitsData changes (e.g., new syllabus uploaded), reset selections
    setSelectedTopics([]);
    setOpenUnitId(null); // Close any open units
  }, [unitsData]);

  // Handle checkbox change for selecting topics for notes generation
  // This now needs to handle both unit titles and sub-unit titles
  const handleCheckboxChange = (topicTitle) => {
    setSelectedTopics((prevSelectedTopics) =>
      prevSelectedTopics.includes(topicTitle)
        ? prevSelectedTopics.filter((title) => title !== topicTitle)
        : [...prevSelectedTopics, topicTitle]
    );
  };

  // Toggle the visibility of a unit's content
  const toggleUnitContent = (unitId) => {
    setOpenUnitId((prevOpenUnitId) => (prevOpenUnitId === unitId ? null : unitId));
  };

  const handleGenerateClick = () => {
    onGenerateNotes(selectedTopics);
  };

  return (
    <div className="bg-white p-8 rounded-2xl shadow-xl border border-gray-100">
      <h2 className="text-3xl font-extrabold text-gray-900 mb-3">
        Select Chapters <span className="text-blue-600">({fileName})</span>
      </h2>
      <p className="text-gray-700 text-lg mb-8 leading-relaxed">
        Choose the units and sub-units you want to generate study notes for. Click on any unit's title to expand and review its detailed content.
      </p>

      {unitsData.length === 0 ? (
        <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-lg flex items-center mb-6">
          <svg className="w-6 h-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>
          </svg>
          <p>No units or chapters could be extracted from the syllabus. Please try another PDF or check its content structure.</p>
        </div>
      ) : (
        <div className="space-y-5">
          {unitsData.map((unit, unitIndex) => (
            <div key={unitIndex} className="bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden hover:shadow-md transition-shadow duration-200">
              {/* Unit Header */}
              <div
                className={`flex items-center justify-between p-5 cursor-pointer select-none ${
                  openUnitId === unitIndex ? 'bg-blue-50 border-b border-blue-200' : 'bg-gray-50 hover:bg-gray-100'
                }`}
                onClick={() => toggleUnitContent(unitIndex)}
              >
                <div className="flex items-center flex-grow">
                  <input
                    type="checkbox"
                    id={`unit-${unitIndex}`}
                    checked={selectedTopics.includes(unit.title)}
                    onChange={() => handleCheckboxChange(unit.title)}
                    className="form-checkbox h-5 w-5 text-blue-600 rounded-md focus:ring-blue-500 transition duration-150 ease-in-out mr-4 flex-shrink-0"
                    onClick={(e) => e.stopPropagation()} // Prevent expansion when clicking checkbox
                  />
                  <label htmlFor={`unit-${unitIndex}`} className="font-semibold text-gray-800 text-xl flex-grow cursor-pointer">
                    {unit.title}
                  </label>
                </div>
                {/* Accordion icon */}
                <svg
                  className={`w-6 h-6 text-gray-500 transform transition-transform duration-300 ${
                    openUnitId === unitIndex ? 'rotate-180 text-blue-600' : 'rotate-0'
                  }`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                </svg>
              </div>

              {/* Collapsible Content - Unit Overview and Sub-units */}
              {openUnitId === unitIndex && (
                <div className="p-6 bg-white">
                  {/* Display unit.content if it exists and is not empty */}
                  {unit.content && unit.content.trim() !== '' && (
                    <div className="mb-4">
                      <h4 className="font-bold text-gray-800 text-lg mb-2">Unit Overview:</h4>
                      <p className="text-gray-700 text-sm leading-relaxed whitespace-pre-wrap">{unit.content}</p>
                    </div>
                  )}

                  {/* Display sub-units if they exist */}
                  {unit.sub_units && unit.sub_units.length > 0 && (
                    <div className="mt-4">
                      <h4 className="font-bold text-gray-800 text-lg mb-3">Sub-Units:</h4>
                      <ul className="space-y-3">
                        {unit.sub_units.map((subUnit, subUnitIndex) => (
                          <li key={subUnitIndex} className="flex items-start bg-gray-50 p-4 rounded-md border border-gray-200">
                            <input
                              type="checkbox"
                              id={`sub-unit-${unitIndex}-${subUnitIndex}`}
                              checked={selectedTopics.includes(subUnit.title)}
                              onChange={() => handleCheckboxChange(subUnit.title)}
                              className="form-checkbox h-4 w-4 text-purple-600 rounded-sm focus:ring-purple-500 transition duration-150 ease-in-out mr-3 flex-shrink-0 mt-1"
                            />
                            <label htmlFor={`sub-unit-${unitIndex}-${subUnitIndex}`} className="flex-grow">
                              <span className="font-medium text-gray-800 text-base">{subUnit.title}</span>
                              {subUnit.content && subUnit.content.trim() !== '' ? (
                                <p className="text-gray-600 text-sm mt-1 leading-snug whitespace-pre-wrap">{subUnit.content}</p>
                              ) : (
                                <p className="text-gray-500 text-xs italic mt-1">No detailed content for this sub-unit.</p>
                              )}
                            </label>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Message if no unit content or sub-units */}
                  {(!unit.content || unit.content.trim() === '') && (!unit.sub_units || unit.sub_units.length === 0) && (
                    <div className="bg-gray-50 border border-gray-200 text-gray-600 p-4 rounded-lg flex items-center">
                      <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                      </svg>
                      <p className="text-sm italic">No detailed content or sub-units found for this unit.</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      <div className="mt-10 flex justify-between items-center border-t border-gray-200 pt-8">
        <button
          onClick={onReset}
          className="px-6 py-3 border border-gray-300 rounded-lg text-gray-700 font-medium hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition duration-150 ease-in-out shadow-sm"
        >
          Upload New Syllabus
        </button>
        <button
          onClick={handleGenerateClick}
          disabled={selectedTopics.length === 0 || isLoadingNotes}
          className={`px-8 py-3 rounded-lg text-white font-semibold transition duration-150 ease-in-out shadow-md ${
            selectedTopics.length === 0 || isLoadingNotes
              ? 'bg-blue-300 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500'
          }`}
        >
          {isLoadingNotes ? 'Generating Notes...' : `Generate Notes (${selectedTopics.length})`}
        </button>
      </div>
    </div>
  );
}

export default ChapterSelection;
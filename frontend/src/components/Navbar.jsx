// src/components/Navbar.jsx
import React from 'react';

const Navbar = () => {
  return (
    <nav className="bg-white shadow-md border-b border-gray-100"> {/* Changed background to white, added shadow and border */}
      <div className="container mx-auto px-6 py-4 flex justify-between items-center"> {/* Increased horizontal padding, slightly more vertical padding */}
        <a href="/" className="text-gray-900 text-3xl font-extrabold flex items-center"> {/* Darker text, larger font, bolder */}
          <svg className="w-9 h-9 mr-2 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"> {/* Slightly larger icon, blue color */}
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"></path>
          </svg>
          NoteMate <span className="text-blue-600 ml-1">AI</span>
        </a>
      </div>
    </nav>
  );
};

export default Navbar;

import React from 'react';
import { StethoscopeIcon } from './icons';

export const Header: React.FC = () => (
  <header className="bg-dark-surface shadow-lg border-b border-gray-700">
    <div className="container mx-auto px-4 py-4 flex items-center justify-center">
        <StethoscopeIcon />
        <h1 className="text-2xl md:text-3xl font-bold ml-3">
          AI X-Ray Medical Report Generator (OFA)
        </h1>
    </div>
  </header>
);

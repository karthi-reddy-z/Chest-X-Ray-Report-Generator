
import React from 'react';

export const Footer: React.FC = () => (
  <footer className="bg-dark-surface mt-8 py-4 border-t border-gray-700">
    <div className="container mx-auto px-4 text-center text-dark-subtext text-sm">
      <p>Powered by OFA Vision Transformer. This is a demonstration tool and not for clinical use.</p>
      <p>&copy; {new Date().getFullYear()} AI Medical Imaging Solutions</p>
    </div>
  </footer>
);

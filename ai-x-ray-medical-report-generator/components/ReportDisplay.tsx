
import React from 'react';
import type { ReportData } from '../types';
import { DownloadIcon } from './icons';

interface ReportDisplayProps {
  reportData: ReportData;
  onDownloadPdf: () => void;
}

export const ReportDisplay: React.FC<ReportDisplayProps> = ({ reportData, onDownloadPdf }) => {
  return (
    <div className="space-y-8">
      <div className="bg-dark-surface rounded-lg p-6 shadow-xl border border-gray-700">
        <h2 className="text-2xl font-bold mb-4 text-center">2. AI Analysis & Visualization</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="text-center">
            <h3 className="font-semibold text-lg mb-2">Original Image</h3>
            <img src={reportData.originalImage} alt="Original X-ray" className="rounded-lg shadow-md w-full border border-gray-600" />
          </div>
          <div className="text-center relative">
            <h3 className="font-semibold text-lg mb-2">AI Analyzed Image (Simulated Grad-CAM)</h3>
            <img src={reportData.analyzedImage} alt="Analyzed X-ray" className="rounded-lg shadow-md w-full border border-gray-600" />
          </div>
        </div>
      </div>

      <div className="bg-dark-surface rounded-lg p-6 shadow-xl border border-gray-700">
        <h2 className="text-2xl font-bold mb-4">3. Generated Textual Report</h2>
        <div className="bg-gray-800/50 p-4 rounded-lg whitespace-pre-wrap text-dark-subtext border border-gray-700">
          {reportData.reportText}
        </div>
        <div className="mt-4 text-sm text-gray-500">
          <p><strong>Model:</strong> {reportData.modelName}</p>
          <p><strong>Timestamp:</strong> {reportData.timestamp}</p>
        </div>
        <div className="text-center mt-6">
          <button
            onClick={onDownloadPdf}
            className="bg-green-600 text-white font-bold py-3 px-8 rounded-lg hover:bg-green-700 transition-colors duration-300 flex items-center justify-center mx-auto"
          >
            <DownloadIcon />
            Download PDF Report
          </button>
        </div>
      </div>
    </div>
  );
};

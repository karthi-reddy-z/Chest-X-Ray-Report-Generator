
import React, { useState, useCallback } from 'react';
import { jsPDF } from 'jspdf';
import html2canvas from 'html2canvas';
import { ImageUploader } from './components/ImageUploader';
import { ReportDisplay } from './components/ReportDisplay';
import { PdfReport } from './components/PdfReport';
import { Header } from './components/Header';
import { Footer } from './components/Footer';
import { generateXrayReport } from './services/geminiService';
import type { ReportData } from './types';
import { createAnalyzedImage } from './utils/imageUtils';
import { LoadingIcon } from './components/icons';

const App: React.FC = () => {
  const [reportData, setReportData] = useState<ReportData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [originalImage, setOriginalImage] = useState<{
    file: File;
    dataUrl: string;
  } | null>(null);

  const handleImageUpload = (file: File, dataUrl: string) => {
    setOriginalImage({ file, dataUrl });
    setReportData(null);
    setError(null);
  };

  const handleAnalyze = useCallback(async () => {
    if (!originalImage) {
      setError('Please upload an image first.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setReportData(null);

    try {
      const { dataUrl } = originalImage;
      const mimeType = originalImage.file.type;
      const base64Data = dataUrl.split(',')[1];
      
      const [generatedText, analyzedImageUrl] = await Promise.all([
        generateXrayReport(base64Data, mimeType),
        createAnalyzedImage(dataUrl),
      ]);

      setReportData({
        originalImage: dataUrl,
        analyzedImage: analyzedImageUrl,
        reportText: generatedText,
        modelName: 'OFA-Sys/ofa-base (Simulated)',
        timestamp: new Date().toLocaleString(),
      });
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : 'An unknown error occurred during analysis.');
    } finally {
      setIsLoading(false);
    }
  }, [originalImage]);
  
  const handleDownloadPdf = async () => {
    if (!reportData) return;

    const pdfContainer = document.getElementById('pdf-report-container');
    if (!pdfContainer) return;
    
    // Temporarily make it visible for rendering
    pdfContainer.style.display = 'block';
    
    const canvas = await html2canvas(pdfContainer, { scale: 2 });
    
    // Hide it again
    pdfContainer.style.display = 'none';

    const imgData = canvas.toDataURL('image/png');
    const pdf = new jsPDF({
      orientation: 'portrait',
      unit: 'px',
      format: 'a4',
    });

    const pdfWidth = pdf.internal.pageSize.getWidth();
    const pdfHeight = pdf.internal.pageSize.getHeight();
    const imgWidth = canvas.width;
    const imgHeight = canvas.height;
    const ratio = Math.min(pdfWidth / imgWidth, pdfHeight / imgHeight);

    const imgX = (pdfWidth - imgWidth * ratio) / 2;
    const imgY = 0;

    pdf.addImage(imgData, 'PNG', imgX, imgY, imgWidth * ratio, imgHeight * ratio);
    pdf.save(`xray-report-${new Date().toISOString().split('T')[0]}.pdf`);
  };

  return (
    <div className="min-h-screen bg-dark-bg text-dark-text flex flex-col font-sans">
      <Header />
      <main className="flex-grow container mx-auto p-4 md:p-8 w-full max-w-6xl">
        <div className="space-y-8">
          <div className="bg-dark-surface rounded-lg p-6 shadow-xl border border-gray-700">
            <h2 className="text-2xl font-bold mb-4 text-center">1. Upload Your X-Ray Image</h2>
            <p className="text-center text-dark-subtext mb-6">Upload a chest X-ray image (JPG or PNG) for analysis.</p>
            <ImageUploader onImageUpload={handleImageUpload} />
          </div>

          {originalImage && (
            <div className="text-center">
              <button
                onClick={handleAnalyze}
                disabled={isLoading}
                className="bg-brand-primary text-white font-bold py-3 px-8 rounded-lg hover:bg-blue-600 transition-colors duration-300 disabled:bg-gray-500 disabled:cursor-not-allowed flex items-center justify-center mx-auto"
              >
                {isLoading ? (
                  <>
                    <LoadingIcon />
                    Analyzing...
                  </>
                ) : (
                  'Generate AI Report'
                )}
              </button>
            </div>
          )}

          {isLoading && (
            <div className="bg-dark-surface rounded-lg p-6 shadow-xl border border-gray-700 flex flex-col items-center justify-center text-center">
                <div className="w-16 h-16 border-4 border-blue-400 border-t-transparent rounded-full animate-spin"></div>
                <p className="mt-4 text-lg font-semibold">AI is analyzing the image...</p>
                <p className="text-dark-subtext">This may take a moment.</p>
            </div>
          )}

          {error && (
             <div className="bg-red-900/50 border border-red-700 text-red-200 px-4 py-3 rounded-lg relative" role="alert">
                <strong className="font-bold">Error: </strong>
                <span className="block sm:inline">{error}</span>
            </div>
          )}

          {reportData && !isLoading && (
            <ReportDisplay 
              reportData={reportData} 
              onDownloadPdf={handleDownloadPdf} 
            />
          )}
        </div>
      </main>
      <Footer />
      <div id="pdf-report-container" className="hidden absolute left-[-9999px] top-0">
          {reportData && <PdfReport reportData={reportData} />}
      </div>
    </div>
  );
};

export default App;


import React, { useState, useCallback } from 'react';
import { UploadIcon } from './icons';

interface ImageUploaderProps {
  onImageUpload: (file: File, dataUrl: string) => void;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({ onImageUpload }) => {
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);

  const handleFileChange = (files: FileList | null) => {
    if (files && files[0]) {
      const file = files[0];
      const reader = new FileReader();
      reader.onloadend = () => {
        const dataUrl = reader.result as string;
        setPreview(dataUrl);
        onImageUpload(file, dataUrl);
      };
      reader.readAsDataURL(file);
    }
  };

  const onDragEnter = useCallback((e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);
  
  const onDragOver = useCallback((e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const onDrop = useCallback((e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileChange(e.dataTransfer.files);
    }
  }, []);


  return (
    <div className="w-full max-w-2xl mx-auto">
      <label
        htmlFor="file-upload"
        onDragEnter={onDragEnter}
        onDragLeave={onDragLeave}
        onDragOver={onDragOver}
        onDrop={onDrop}
        className={`flex flex-col items-center justify-center w-full h-64 border-2 border-dashed rounded-lg cursor-pointer transition-colors duration-300
          ${isDragging ? 'border-brand-primary bg-blue-900/20' : 'border-gray-600 bg-gray-800/50 hover:border-gray-500 hover:bg-gray-800'}`}
      >
        <div className="flex flex-col items-center justify-center pt-5 pb-6">
          <UploadIcon />
          <p className="mb-2 text-sm text-dark-subtext"><span className="font-semibold text-dark-text">Click to upload</span> or drag and drop</p>
          <p className="text-xs text-dark-subtext">PNG or JPG</p>
        </div>
        <input id="file-upload" type="file" className="hidden" accept="image/png, image/jpeg" onChange={(e) => handleFileChange(e.target.files)} />
      </label>

      {preview && (
        <div className="mt-6 text-center">
          <h3 className="text-lg font-semibold mb-2">Image Preview</h3>
          <img src={preview} alt="X-ray preview" className="max-w-xs max-h-64 mx-auto rounded-lg shadow-lg border border-gray-700" />
        </div>
      )}
    </div>
  );
};

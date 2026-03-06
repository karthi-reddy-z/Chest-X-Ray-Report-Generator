
import { GoogleGenerativeAI } from '@google/generative-ai';

export async function generateXrayReport(base64Data: string, mimeType: string): Promise<string> {

  const image = {
    inlineData: {
      data: base64Data,
      mimeType,
    },
  };

  const result = await model.generateContent(['Generate a detailed X-ray report for the given image. Format the report using numbers, hashes, and asterisks for headings and bullet points, similar to a markdown format. For example: ## CHEST X-RAY REPORT, **STUDY:**, 1. FINDINGS.', image]);
  const response = await result.response;
  const text = response.text();
  
  return text;
}

const API_KEY = import.meta.env.VITE_GEMINI_API_KEY;

const genAI = new GoogleGenerativeAI(API_KEY);

const model = genAI.getGenerativeModel({ model: 'gemini-flash-latest' });

// console.log('Simulating OFA model inference for image with mimeType:', mimeType, 'and data length:', base64Data.length);

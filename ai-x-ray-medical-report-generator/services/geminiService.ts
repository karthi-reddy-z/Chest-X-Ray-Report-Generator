import { GoogleGenerativeAI } from '@google/generative-ai';

const API_KEY = import.meta.env.VITE_GEMINI_API_KEY;

const genAI = new GoogleGenerativeAI(API_KEY);

const model = genAI.getGenerativeModel({
  model: 'gemini-1.5-flash',
});

export async function generateXrayReport(
  base64Data: string,
  mimeType: string
): Promise<string> {

  const prompt = `
You are an expert radiologist AI.

Analyze the chest X-ray image and produce a report EXACTLY in the format below.

Return the report ONLY in this format.

## AI CHEST X-RAY REPORT

### STUDY INFORMATION
Imaging Type:
Projection:

### IMAGE QUALITY
Exposure:
Rotation:
Artifacts:

### ANATOMICAL FINDINGS
Lungs:
Pleura:
Mediastinum:
Cardiac Silhouette:
Diaphragm:
Bones:

### AI DISEASE SCREENING
Pneumonia: (Detected / Not Detected) - Confidence %
Tuberculosis: (Detected / Not Detected) - Confidence %
Lung Nodule: (Detected / Not Detected) - Confidence %
Pleural Effusion: (Detected / Not Detected) - Confidence %

### IMPRESSION
1.
2.
3.

### AI CONFIDENCE
Overall Diagnostic Confidence: %

Rules:
- Always fill every section.
- Use professional radiology terminology.
- If nothing abnormal is visible, state "No abnormality detected".
`;

  try {

    const result = await model.generateContent({
      contents: [
        {
          role: "user",
          parts: [
            { text: prompt },
            {
              inlineData: {
                mimeType: mimeType,
                data: base64Data
              }
            }
          ]
        }
      ]
    });

    const text = result.response.text();

    return text;

  } catch (error) {

    console.error("AI report generation failed:", error);

    return "Error generating report. Please try again.";

  }
}

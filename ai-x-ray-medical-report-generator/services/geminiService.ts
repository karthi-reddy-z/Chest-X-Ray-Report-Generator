import { GoogleGenerativeAI } from '@google/generative-ai';

const API_KEY = import.meta.env.VITE_GEMINI_API_KEY;

const genAI = new GoogleGenerativeAI(API_KEY);

// Use stable Gemini model
const model = genAI.getGenerativeModel({
  model: 'gemini-1.5-flash',
});

export async function generateXrayReport(
  base64Data: string,
  mimeType: string
): Promise<string> {

  const image = {
    inlineData: {
      data: base64Data,
      mimeType: mimeType,
    },
  };

  const prompt = `
You are an expert radiologist AI.

Analyze the provided chest X-ray image and generate a professional structured radiology report.

Follow this format strictly:

## CHEST X-RAY REPORT

### STUDY
- Imaging Type: Chest X-ray
- Projection (PA/AP if visible)

### IMAGE QUALITY
- Exposure quality
- Rotation
- Artifacts if present

### FINDINGS

**Lungs**
- Evaluate lung fields
- Look for consolidation, nodules, or opacities

**Pleura**
- Check for pleural effusion
- Check for pneumothorax

**Mediastinum**
- Evaluate mediastinal contours
- Check trachea position

**Cardiac Silhouette**
- Assess heart size
- Cardiothoracic ratio estimation

**Diaphragm**
- Evaluate diaphragmatic contour
- Presence of gastric bubble

**Bones**
- Inspect ribs, clavicles, and spine
- Look for fractures or lesions

### AI DISEASE SCREENING

Provide detection status and confidence percentage.

- Pneumonia: Detected / Not Detected (Confidence %)
- Tuberculosis: Detected / Not Detected (Confidence %)
- Lung Nodule: Detected / Not Detected (Confidence %)
- Pleural Effusion: Detected / Not Detected (Confidence %)

### IMPRESSION
Provide 2–3 concise diagnostic conclusions.

### AI CONFIDENCE SCORE
Provide an overall diagnostic confidence percentage.

Use professional radiology terminology and bullet points.
Avoid hallucinating diseases if not visible.
`;

  try {

    const result = await model.generateContent([prompt, image]);

    const response = await result.response;

    const text = response.text();

    return text;

  } catch (error) {

    console.error("AI report generation failed:", error);

    return "Error generating report. Please try again.";

  }
}

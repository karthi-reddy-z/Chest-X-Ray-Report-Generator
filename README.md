# 🩻 Chest X-Ray Report Generator

An AI-powered web application that automatically generates **medical reports from chest X-ray images** using a **Vision-Language Transformer (OFA) architecture**. The system analyzes uploaded X-ray images and produces structured radiology reports to assist healthcare professionals.

---
🌐 Live Demo

https://chest-x-ray-report-generator.vercel.app/
---
# 🚀 Features
- Upload chest X-ray images
- AI-generated radiology reports
- Vision-language model using **OFA Transformer**
- **CheXNet** for visual feature extraction
- Download generated reports as **PDF**
- Interactive **React + TypeScript** interface

---

# 📊 Dataset
The model uses the **IU-XRay dataset**, which contains chest X-ray images paired with expert-written radiology reports.

---

# 🛠 Tech Stack

### Frontend
- React  
- TypeScript  
- Vite  

### AI / Machine Learning
- OFA Transformer  
- CheXNet (Pretrained Visual Model)  
- Medical Word Embeddings  

### Backend Services
- Gemini API for report processing  

### Other Tools
- PDF generation  
- Medical dataset preprocessing  

---

# 🗂 Project Structure

```
ai-x-ray-medical-report-generator
│
├── IU-XRay
│   ├── images
│   ├── all_data.csv
│   ├── training_set.csv
│   └── testing_set.csv
│
├── components
│   ├── Header.tsx
│   ├── Footer.tsx
│   ├── ImageUploader.tsx
│   ├── ReportDisplay.tsx
│   ├── PdfReport.tsx
│   └── icons.tsx
│
├── medical_word_embeddings
│   └── saved_embeddings.pickle
│
├── pretrained_visual_model
│   ├── fine_tuned_chexnet.h5
│   └── fine_tuned_chexnet.json
│
├── services
│   └── geminiService.ts
│
├── utils
│
├── App.tsx
├── ofa.tsx
├── index.tsx
├── types.ts
├── metadata.json
├── package.json
└── vite.config.ts
```


---

# ▶️ Installation

### Clone the repository

```
git clone https://github.com/karthi-reddy-z/Chest-X-Ray-Report-Generator.git
```
Navigate to the project folder
```
cd Chest-X-Ray-Report-Generator
```
Install dependencies
```
npm install
```
Run the project
```
npm run dev
```

---
👨‍💻 Team 

M. Venkata Karthik Reddy

S. Malakonda

G. Mahesh

B. Sai Koti Reddy

---
📜 License

This project is intended for academic and research purposes.

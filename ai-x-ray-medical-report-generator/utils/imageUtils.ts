
// Helper function to generate a random number within a range
const getRandom = (min: number, max: number) => Math.random() * (max - min) + min;

export const createAnalyzedImage = (imageUrl: string): Promise<string> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        return reject(new Error('Could not get canvas context'));
      }

      canvas.width = img.width;
      canvas.height = img.height;

      // Draw the original image
      ctx.drawImage(img, 0, 0);

      // Create a more realistic simulated Grad-CAM overlay with multiple hotspots
      ctx.globalCompositeOperation = 'overlay';

      const hotspotCount = Math.floor(getRandom(2, 5)); // Generate 2 to 4 hotspots
      for (let i = 0; i < hotspotCount; i++) {
        const centerX = getRandom(canvas.width * 0.25, canvas.width * 0.75);
        const centerY = getRandom(canvas.height * 0.25, canvas.height * 0.75);
        const radius = getRandom(Math.min(canvas.width, canvas.height) * 0.15, Math.min(canvas.width, canvas.height) * 0.3);
        
        const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius);
        gradient.addColorStop(0, `rgba(255, 0, 0, ${getRandom(0.5, 0.7)})`);
        gradient.addColorStop(0.5, `rgba(255, 255, 0, ${getRandom(0.4, 0.6)})`);
        gradient.addColorStop(1, 'rgba(0, 255, 0, 0)');
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
        ctx.fill();
      }
      
      ctx.globalCompositeOperation = 'source-over'; // Reset composite operation

      resolve(canvas.toDataURL('image/png'));
    };
    img.onerror = (err) => {
      reject(err);
    };
    img.src = imageUrl;
  });
};

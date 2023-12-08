import React, { useRef } from 'react';
import Button from '@mui/material/Button';

const ImageCaptureComponent = () => {
  const videoRef = useRef(null);

  const captureImage = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
    }
  };
  
  const uploadCapturedImage = async () => {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
  
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
  
    context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
  
    // Extract only the base64-encoded image data
    const imageData = canvas.toDataURL('image/png').split(',')[1]; 
  
    // Now you can upload `imageData` to the backend using fetch
    try {
      const response = await fetch('http://localhost:5000/api/uploadImage', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `image_data=${encodeURIComponent(imageData)}`,
      });
      console.log('Response Status:', response.status);
  
      const result = await response.json();
      console.log(result);
    } catch (error) {
      console.error('Error uploading image:', error);
    }
  };
  
  return (
<div>
    <Button variant="contained" color="primary" onClick={captureImage}>
      Start Image Capture
    </Button>
    <Button variant="contained" color="secondary" onClick={uploadCapturedImage}>
      Upload Captured Image
    </Button>
    <video ref={videoRef} width="400" height="300" autoPlay muted />
  </div>
  );
};

export default ImageCaptureComponent;

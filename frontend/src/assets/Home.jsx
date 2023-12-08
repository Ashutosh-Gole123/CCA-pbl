// import React from 'react'

import { Box, Button, FormControl, Input, InputLabel, Modal, TextField, Typography } from "@mui/material";
import Header from "./Header";
import ArrowCircleRightIcon from "@mui/icons-material/ArrowCircleRight";
import { Reveal } from "./animates/Reveal";
import img from "./images/img.jpg";
import { useState } from "react";
import ImageCaptureComponent from "./Capture";

const style = {
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  width: 400,
  bgcolor: 'background.paper',
  border: '2px solid #000',
  boxShadow: 24,
  p: 4,
};

function Home() {
  const [file, setFile] = useState(null);
  const [open, setOpen] = useState(false);
  const handleOpen = () => setOpen(true);
  const handleClose = () => setOpen(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    handleUpload()
  };

  const handleUpload = () => {
    if (file) {
      const formData = new FormData();
      formData.append('photo_data', file);

      fetch('http://localhost:5000/api/upload', {
        method: 'POST',
        body: formData,
      })
        .then((response) => response.json())
        .then((data) => {
          console.log(data);
        })
        .catch((error) => {
          console.error('Error uploading file:', error);
        });
    }
  };
  return (
    <>
      <div style={{ backgroundColor: "#CCDAE0", height: "80vh" }}>
        <Header />
        <div
          style={{
            margin: "0 20px",
            width: "100%",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            textAlign: "center",
            height: "80vh",
          }}
        >
          <div
            style={{
              margin: "0 20px",
              width: "50%",
              display: "flex",
              flexDirection: "column",
              justifyContent: "center",
              alignItems: "center",
              textAlign: "center",
              height: "80vh",
            }}
          >
            <Reveal>
              <Typography component="h2" variant="h4">
                Welcome to the world of learning! Here you can find a variety of
              </Typography>
            </Reveal>
            <Reveal>
            

        <Button
          variant="contained"
          color="success"
          endIcon={<ArrowCircleRightIcon />}
          component="span" 
          onClick={handleOpen}
          // This is important for the label to work as expected
        >
          Upload photo
        </Button>
            </Reveal>
      <Modal
        open={open}
        onClose={handleClose}
        aria-labelledby="modal-modal-title"
        aria-describedby="modal-modal-description"
      >
        <Box sx={style}>
        <FormControl fullWidth>
      <Input
        id="outlined-file-input"
        type="file"
        onChange={handleFileChange}
        inputProps={{ accept: 'image/*' }} // Specify accepted file types if needed
      />
         <Button
          variant="contained"
          color="success"
          endIcon={<ArrowCircleRightIcon />}
          component="span" 
          onClick={handleUpload}
          // This is important for the label to work as expected
        >
          Upload
        </Button>
        
    </FormControl>
        </Box>
      </Modal>
      <ImageCaptureComponent/>
          </div>
          {/* <img src={img} alt="" style={{ maxWidth: "100%" }} /> */}
        </div>
      </div>
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 320">
        <path
          fill="#CCDAE0"
          fill-opacity="1"
          d="M0,160L48,149.3C96,139,192,117,288,96C384,75,480,53,576,58.7C672,64,768,96,864,106.7C960,117,1056,107,1152,90.7C1248,75,1344,53,1392,42.7L1440,32L1440,0L1392,0C1344,0,1248,0,1152,0C1056,0,960,0,864,0C768,0,672,0,576,0C480,0,384,0,288,0C192,0,96,0,48,0L0,0Z"
        ></path>
      </svg>
    </>
  );
}

export default Home;

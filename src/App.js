import React, { useRef, useEffect } from "react";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";
import * as poseDetection from "@tensorflow-models/pose-detection";
import Webcam from "react-webcam";
import "./App.css";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const videoConstraints = {
  width: 640,
  height: 480,
  facingMode: "user" // Forces the front/selfie camera
};

    const runPoseDetection = async () => {
    // 1. ADD THIS LINE: Wait for TFJS to be ready
    await tf.ready();

    // Optional: Explicitly set the backend to WebGL if WebGPU is acting up
    await tf.setBackend('webgl');

    const detector = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
    );

    setInterval(() => {
      detect(detector);
    }, 100);
  };

  const detect = async (detector) => {
  if (webcamRef.current && webcamRef.current.video.readyState === 4) {
    const video = webcamRef.current.video;
    const { videoWidth, videoHeight } = video;

    // Set internal canvas resolution to match video stream
    if (canvasRef.current.width !== videoWidth) {
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;
    }

    const poses = await detector.estimatePoses(video);
    const ctx = canvasRef.current.getContext("2d");
    
    // Call your draw function
    drawCanvas(poses, ctx);
  }
};


  const drawCanvas = (poses, ctx) => {
  ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

  poses.forEach(({ keypoints }) => {
    keypoints.forEach((kp) => {
      // Lower this to 0.2 temporarily to see if dots appear
      if (kp.score > 0.5) { 
        ctx.beginPath();
        ctx.arc(kp.x, kp.y, 8, 0, 2 * Math.PI);
        ctx.fillStyle = "aqua";
        ctx.fill();
      }
    });
  });
};

  useEffect(() => { runPoseDetection(); }, []);

  return (
  <div style={{ 
    position: "relative", 
    width: "640px", 
    height: "480px", 
    margin: "auto",
    border: "2px solid red" // This helps you see the outer container
  }}>
    <Webcam
  ref={webcamRef}
  videoConstraints={videoConstraints}
  onUserMediaError={(err) => console.error("Webcam Error: ", err)}
  style={{ position: "absolute", zIndex: 1 }}
/>
    <canvas
      ref={canvasRef}
      style={{
        position: "absolute",
        left: 0,
        top: 0,
        width: 640,
        height: 480,
        zIndex: 10,
        pointerEvents: "none"
      }}
    />
  </div>
);
}

export default App;

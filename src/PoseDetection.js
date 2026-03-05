import React, { useRef, useEffect } from "react";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";
import * as poseDetection from "@tensorflow-models/pose-detection";
import Webcam from "react-webcam";

const PoseDetection = () => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  const runPoseDetection = async () => {
    // 1. Load the MoveNet model (Lightning version for speed)
    const detectorConfig = {
      modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
    };
    const detector = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      detectorConfig
    );

    // 2. Start the detection loop
    setInterval(() => {
      detect(detector);
    }, 100);
  };

  const detect = async (detector) => {
    if (
      webcamRef.current &&
      webcamRef.current.video.readyState === 4
    ) {
      const video = webcamRef.current.video;
      const { videoWidth, videoHeight } = video;

      // Set video and canvas dimensions
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      // 3. Estimate Poses
      const poses = await detector.estimatePoses(video);
      
      // 4. Draw to Canvas
      drawCanvas(poses, videoWidth, videoHeight, canvasRef.current.getContext("2d"));
    }
  };

  const drawCanvas = (poses, width, height, ctx) => {
    ctx.clearRect(0, 0, width, height);
    poses.forEach(({ keypoints }) => {
      keypoints.forEach((keypoint) => {
        if (keypoint.score > 0.5) { // Only draw confident points
          const { x, y } = keypoint;
          ctx.beginPath();
          ctx.arc(x, y, 5, 0, 2 * Math.PI);
          ctx.fillStyle = "red";
          ctx.fill();
        }
      });
    });
  };

  useEffect(() => { runPoseDetection(); }, []);

  return (
    <div style={{ position: "relative" }}>
      <Webcam
        ref={webcamRef}
        style={{ position: "absolute", left: 0, top: 0, width: 640, height: 480 }}
      />
      <canvas
        ref={canvasRef}
        style={{ position: "absolute", left: 0, top: 0, width: 640, height: 480 }}
      />
    </div>
  );
};

export default PoseDetection;

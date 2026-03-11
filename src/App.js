import React, { useRef, useEffect } from "react";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";
import * as poseDetection from "@tensorflow-models/pose-detection";
import Webcam from "react-webcam";
import "./App.css";

function App() {
	const webcamRef = useRef(null);
	const canvasRef = useRef(null);
	const lastPosesRef = useRef(null);
	const videoConstraints = {
		width: 640,
		height: 480,
		facingMode: "user" // Forces the front/selfie camera
	};
	const SKELETON_CONNECTIONS = [
		// Center (Face & Shoulders) - Yellow
		{ pairs: [0, 1], color: "yellow" }, { pairs: [0, 2], color: "yellow" },
		{ pairs: [1, 3], color: "yellow" }, { pairs: [2, 4], color: "yellow" },
		{ pairs: [5, 6], color: "yellow" }, { pairs: [11, 12], color: "yellow" },
		{ pairs: [5, 11], color: "yellow" }, { pairs: [6, 12], color: "yellow" },

		// Left Side (Arms & Legs) - Blue
		{ pairs: [5, 7], color: "cyan" }, { pairs: [7, 9], color: "cyan" },   // Arm
		{ pairs: [11, 13], color: "cyan" }, { pairs: [13, 15], color: "cyan" }, // Leg

		// Right Side (Arms & Legs) - Red/Orange
		{ pairs: [6, 8], color: "orange" }, { pairs: [8, 10], color: "orange" }, // Arm
		{ pairs: [12, 14], color: "orange" }, { pairs: [14, 16], color: "orange" } // Leg
	];


	const runPoseDetection = async () => {
		await tf.ready();
		//await tf.setBackend('webgl');

		const detectorConfig = {
			// can switch between LIGHTNING and THUNDER for different speed/accuracy tradeoffs 
			// (lightning is faster but less accurate, thunder is slower but more accurate)
			modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER,
			enableSmoothing: true // Enables built-in jitter reduction
		};

		const detector = await poseDetection.createDetector(
			poseDetection.SupportedModels.MoveNet,
			detectorConfig
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

			if (poses.length > 0) {
				const smoothedPoses = poses.map(pose => {
					// If we have a previous pose, blend them
					if (lastPosesRef.current) {
						const prevKeypoints = lastPosesRef.current[0].keypoints;
						const alpha = 0.6; // 0.1 = heavy smoothing (laggy), 0.9 = light smoothing (jittery)

						pose.keypoints = pose.keypoints.map((kp, i) => ({
							...kp,
							x: kp.x * alpha + prevKeypoints[i].x * (1 - alpha),
							y: kp.y * alpha + prevKeypoints[i].y * (1 - alpha)
						}));
					}
					return pose;
				});

				lastPosesRef.current = smoothedPoses; // Store for next frame

				// Get the canvas context before drawing
				const ctx = canvasRef.current.getContext("2d");
				drawCanvas(smoothedPoses, ctx);
			}
		}
	};


	const drawCanvas = (poses, ctx) => {
		ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

		poses.forEach(({ keypoints }) => {
			// 1. Draw the Colored Lines (Bones)
			SKELETON_CONNECTIONS.forEach((conn) => {
				const kp1 = keypoints[conn.pairs[0]];
				const kp2 = keypoints[conn.pairs[1]];

				if (kp1.score > 0.3 && kp2.score > 0.3) {
					ctx.beginPath();
					ctx.lineWidth = 4;
					ctx.strokeStyle = conn.color;
					ctx.lineCap = "round";
					ctx.moveTo(kp1.x, kp1.y);
					ctx.lineTo(kp2.x, kp2.y);
					ctx.stroke();
				}
			});

			// 2. Draw the Dots (Joints)
			keypoints.forEach((kp) => {
				if (kp.score > 0.3) {
					ctx.beginPath();
					ctx.arc(kp.x, kp.y, 4, 0, 2 * Math.PI);
					ctx.fillStyle = "white";
					ctx.fill();
					// Optional: add a small border to dots for visibility
					ctx.strokeStyle = "black";
					ctx.lineWidth = 1;
					ctx.stroke();
				}
			});
		});
	};

	useEffect(() => { runPoseDetection(); }, []);

	return (
		<div className="App">
			<div className="detection-container">
				<Webcam
					ref={webcamRef}
					videoConstraints={videoConstraints}
					className="webcam-style"
					onUserMediaError={(err) => console.error("Webcam Error: ", err)}
				/>
				<canvas
					ref={canvasRef}
					className="canvas-style"
				/>
			</div>
		</div>
	);
}

export default App;

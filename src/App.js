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

	const SKELETON_TREE = [
		// 1. Starting at Mid-Hip (0,0) - Branch Up for Torso
		{ id: 11, parent: 'root', side: -1, len: 20, type: 'hip_width' }, // L Hip
		{ id: 12, parent: 'root', side: 1, len: 20, type: 'hip_width' }, // R Hip

		// 2. Spine/Torso (Mid-Hip to Mid-Shoulder)
		{ id: 'mid_shoulder', parent: 'root', side: 1, len: 90, type: 'spine', angle: 180 },

		// 3. Shoulders (From Mid-Shoulder)
		{ id: 5, parent: 'mid_shoulder', side: -1, len: 30, type: 'sh_width', angle: 90 }, // L Shoulder
		{ id: 6, parent: 'mid_shoulder', side: 1, len: 30, type: 'sh_width', angle: 90 }, // R Shoulder

		// 4. Arms (From Shoulders)
		{ id: 7, parent: 5, side: -1, len: 50, type: 'limb' }, // L Elbow
		{ id: 9, parent: 7, side: -1, len: 40, type: 'limb' }, // L Wrist
		{ id: 8, parent: 6, side: 1, len: 50, type: 'limb' }, // R Elbow
		{ id: 10, parent: 8, side: 1, len: 40, type: 'limb' }, // R Wrist

		// 5. Legs (From Hips)
		{ id: 13, parent: 11, side: -1, len: 50, type: 'limb' }, // L Knee
		{ id: 15, parent: 13, side: -1, len: 45, type: 'limb' }, // L Ankle
		{ id: 14, parent: 12, side: 1, len: 50, type: 'limb' }, // R Knee
		{ id: 16, parent: 14, side: 1, len: 45, type: 'limb' }  // R Ankle
	  ];

	const JOINT_MAP = {
		leftElbow: [6, 8, 10],      // Shoulder, Elbow, Wrist
		rightElbow: [5, 7, 9],    // Shoulder, Elbow, Wrist
		leftShoulder: [12, 6, 8],  // Hip, Shoulder, Elbow
		rightShoulder: [11, 5, 7], // Hip, Shoulder, Elbow
		leftKnee: [12, 14, 16],     // Hip, Knee, Ankle
		rightKnee: [11, 13, 15],    // Hip, Knee, Ankle
		leftHip: [6, 12, 14], 	  // Shoulder, Hip, Knee
		rightHip: [5, 11, 13] 	  // Shoulder, Hip, Knee
	};

	// Define coordinates for a "Guide Pose" (e.g., Arms out like a 'T')
	const POSES = {
		T_POSE: {
			name: "T-Pose",
			targetAngles: {
				leftShoulder: 90,  // Arm at 90 degrees (midway between down=0 and up=180, i.e., out to side)
				rightShoulder: 90, // Arm at 90 degrees (midway between down=0 and up=180, i.e., out to side)
				leftElbow: 180,   // Straight arm (180 = straight)
				rightElbow: 180,   // Straight arm (180 = straight)
				leftKnee: 180,        // Leg straight down (180 = down)
				rightKnee: 180,       // Leg straight down (180 = down)
				leftHip: 180,       // Leg straight down (180 = down)
				rightHip: 180,       // Leg straight down (180 = down)
			}
		},
		HIGH_KNEE: {
			name: "High Knee",
			targetAngles: {
				leftShoulder: 25,  // Arm mostly down (25 degrees from down=0)
				rightShoulder: 25, // Arm mostly down (25 degrees from down=0)
				leftElbow: 170,     // Arm slightly bent (170 < 180 straight)
				rightElbow: 170,    // Arm slightly bent (170 < 180 straight)
				leftKnee: 90,        // Knee bent (90 degrees, midway between up=0 and down=180)
				rightKnee: 180,       // Leg straight down (180 = down)
				leftHip: 90,       // Hip bent up (90 degrees, midway between up=0 and down=180)
				rightHip: 180,       // Leg straight down (180 = down)
			}
		  },
		LEFT_ARM_UP: {
			name: "Left Arm Up",
			targetAngles: {
				leftShoulder: 180,  // Arm pointed up (180 = up)
				rightShoulder: 25, // Arm mostly down (25 degrees from down=0)
				leftElbow: 180,     // Straight arm (180 = straight)
				rightElbow: 170,    // Arm slightly bent (170 < 180 straight)
				leftKnee: 180,        // Leg straight down (180 = down)
				rightKnee: 180,       // Leg straight down (180 = down)
				leftHip: 180,       // Leg straight down (180 = down)
				rightHip: 180,       // Leg straight down (180 = down)
			}
		  },
		BOTH_ARM_UP: {
			name: "Both Arms Up",
			targetAngles: {
				leftShoulder: 180,  // Arm pointed up (180 = up)
				rightShoulder: 180, // Arm pointed up (180 = up)
				leftElbow: 180,     // Straight arm (180 = straight)
				rightElbow: 180,    // Straight arm (180 = straight)
				leftKnee: 180,        // Leg straight down (180 = down)
				rightKnee: 180,       // Leg straight down (180 = down)
				leftHip: 180,       // Leg straight down (180 = down)
				rightHip: 180,       // Leg straight down (180 = down)
			}
		},
		REST: {
			name: "Rest Pose",
			targetAngles: {
				leftShoulder: 25,  // Arm mostly down (25 degrees from down=0)
				rightShoulder: 25, // Arm mostly down (25 degrees from down=0)
				leftElbow: 170,    // Arm slightly bent (170 < 180 straight)
				rightElbow: 170,   // Arm slightly bent (170 < 180 straight)
				leftKnee: 180,        // Leg straight down (180 = down)
				rightKnee: 180,       // Leg straight down (180 = down)
				leftHip: 180,       // Leg straight down (180 = down)
				rightHip: 180,       // Leg straight down (180 = down)
			}
		},
	  };



	const [currentPoseName, setCurrentPoseName] = React.useState("REST");
	const [score, setScore] = React.useState(0);

	const getAngle = (p1, p2, p3) => {
		const rad = Math.atan2(p3.y - p2.y, p3.x - p2.x) - Math.atan2(p1.y - p2.y, p1.x - p2.x);
		let angle = Math.abs((rad * 180) / Math.PI);
		if (angle > 180) angle = 360 - angle;
		return angle;
	  };

	const calculateOffsetsFromAngles = (targetAngles) => {
		const offsets = { root: { x: 0, y: 0 } };
		const getLen = (id) => SKELETON_TREE.find(n => n.id === id)?.len || 0;

		// Torso
		offsets['mid_shoulder'] = { x: 0, y: -getLen('mid_shoulder') };
		offsets[5] = { x: offsets['mid_shoulder'].x - getLen(5), y: offsets['mid_shoulder'].y };
		offsets[6] = { x: offsets['mid_shoulder'].x + getLen(6), y: offsets['mid_shoulder'].y };
		offsets[11] = { x: -getLen(11), y: 0 };
		offsets[12] = { x: getLen(12), y: 0 };

		// ARMS
		const processArm = (side, shId, elId, wrId, shS, elS) => {
			const shDeg = targetAngles[shS] ?? 180;
			const elDeg = targetAngles[elS] ?? 180;
			const shRad = shDeg * (Math.PI / 180);
			const elRad = shRad + ((elDeg - 180) * (Math.PI / 180));

			offsets[elId] = {
				x: offsets[shId].x + (side * getLen(elId) * Math.abs(Math.sin(shRad))),
				y: offsets[shId].y + (getLen(elId) * Math.cos(shRad))
			};
			offsets[wrId] = {
				x: offsets[elId].x + (side * getLen(wrId) * Math.abs(Math.sin(elRad))),
				y: offsets[elId].y + (getLen(wrId) * Math.cos(elRad))
			};
		};

		// LEGS
		const processLeg = (side, hipId, knId, akId, hipS, knS) => {
			const hipDeg = targetAngles[hipS] ?? 180;
			const knDeg = targetAngles[knS] ?? 180;

			const hipRad = hipDeg * (Math.PI / 180);

			// Calculate Knee Position
			offsets[knId] = {
				x: offsets[hipId].x + (side * getLen(knId) * Math.abs(Math.sin(hipRad))),
				y: offsets[hipId].y - (getLen(knId) * Math.cos(hipRad))
			};

			// KNEE LOGIC: 
			// We calculate the ankle angle, but we force the Y-direction 
			// to be 'Down' relative to the knee joint.
			const akRad = hipRad + ((knDeg - 180) * (Math.PI / 180));

			offsets[akId] = {
				x: offsets[knId].x + (side * getLen(akId) * Math.abs(Math.sin(akRad))),
				// Use Math.abs(cos) and ensure we ADD to y to move DOWN the canvas
				y: offsets[knId].y + Math.abs(getLen(akId) * Math.cos(akRad))
			};
		};

		// Execute
		processArm(-1, 5, 7, 9, 'leftShoulder', 'leftElbow');
		processArm(1, 6, 8, 10, 'rightShoulder', 'rightElbow');
		processLeg(-1, 11, 13, 15, 'leftHip', 'leftKnee');
		processLeg(1, 12, 14, 16, 'rightHip', 'rightKnee');

		return offsets;
	};

	const calculatePoseScore = (keypoints, targetPoseData) => {
		if (!targetPoseData?.targetAngles) return 0;

		// after calling getVisibleLimbCount, 
		// check if there is atleast 2 visible limbs otherwise return 0 score to prevent false positives.
		const visibleLimbs = getVisibleLimbCount(keypoints);
		if (visibleLimbs < 2) return 0;

		const scores = [];
		const targets = targetPoseData.targetAngles;

		// Iterate over every joint defined in the target pose
		Object.keys(targets).forEach((jointName) => {
			const indices = JOINT_MAP[jointName];
			if (!indices) return; // Skip if joint isn't mapped

			const [p1, p2, p3] = indices.map(idx => keypoints[idx]);

			// Dynamic Visibility Check
			if (p1?.score > 0.3 && p2?.score > 0.3 && p3?.score > 0.3) {
				const currentAngle = getAngle(p1, p2, p3);
				const targetAngle = targets[jointName];

				// Calculate error (0 to 100)
				const diff = Math.abs(currentAngle - targetAngle);
				const jointScore = Math.max(0, 100 - diff);

				scores.push(jointScore);
			}
		});

		if (scores.length === 0) return 0;

		// Return average of all detected/required joints
		return parseFloat((scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(1));
	  };

	const getVisibleLimbCount = (keypoints) => {
		let visibleLimbs = 0;
		const threshold = 0.5;

		// Define the pairs that make a "limb"
		const limbs = [
			['left_shoulder', 'left_elbow'],
			['left_elbow', 'left_wrist'],
			['right_shoulder', 'right_elbow'],
			['right_elbow', 'right_wrist'],
			['left_hip', 'left_knee'],
			['left_knee', 'left_ankle'],
			['right_hip', 'right_knee'],
			['right_knee', 'right_ankle']
		];

		limbs.forEach(([jointA, jointB]) => {
			const kpA = keypoints.find(k => k.name === jointA);
			const kpB = keypoints.find(k => k.name === jointB);

			// If both ends of the bone are clearly seen, count it
			if (kpA?.score > threshold && kpB?.score > threshold) {
				visibleLimbs++;
			}
		});

		return visibleLimbs;
	};

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

				// Calculate score based on available keypoints
				const activePose = POSES[currentPoseName];
				const currentScore = calculatePoseScore(smoothedPoses[0].keypoints, activePose);
				setScore(currentScore);
				//console.log(`Current Pose Score: ${currentScore}%`);

				// Get the canvas context before drawing
				const ctx = canvasRef.current.getContext("2d");
				drawCanvas(smoothedPoses, ctx, currentScore);
			}
		}
	};

	const drawAdaptiveGuide = (ctx, userPose, targetPoseData) => {
		if (!userPose) return;

		const keypoints = userPose.keypoints;
		const leftS = userPose.keypoints[11];
		const rightS = userPose.keypoints[12];

		// Find the User's "Center" and "Scale"
		const centerX = (leftS.x + rightS.x) / 2;
		const centerY = (leftS.y + rightS.y) / 2;
		const userScale = Math.sqrt(
			Math.pow(leftS.x - rightS.x, 2) + Math.pow(leftS.y - rightS.y, 2)
		)/40; // Normalized scale based on shoulder width

		ctx.save();
		ctx.globalAlpha = 0.7;
		ctx.setLineDash([5, 5]);
		ctx.strokeStyle = "white";
		ctx.lineWidth = 10;

		// Draw the skeleton based on the offsets scaled to the user
		SKELETON_CONNECTIONS.forEach((conn) => {
			const p1Idx = conn.pairs[0];
			const p2Idx = conn.pairs[1];
			const userKp1 = keypoints[p1Idx];
			const userKp2 = keypoints[p2Idx];
			const off1 = targetPoseData.offsets[conn.pairs[0]];
			const off2 = targetPoseData.offsets[conn.pairs[1]];

			if (off1 && off2 && userKp1.score > 0.3 && userKp2.score > 0.3) {
				ctx.beginPath();
				ctx.moveTo(centerX + off1.x * userScale, centerY + off1.y * userScale);
				ctx.lineTo(centerX + off2.x * userScale, centerY + off2.y * userScale);
				ctx.stroke();
			}
		});

		Object.keys(targetPoseData.offsets).forEach((idx) => {
			const off = targetPoseData.offsets[idx];
			const userKp = keypoints[idx];

			// ONLY draw the guide dot if the user's joint is visible
			if (userKp && userKp.score > 0.3) {
				ctx.beginPath();
				ctx.arc(centerX + off.x * userScale, centerY + off.y * userScale, 5, 0, 2 * Math.PI);
				ctx.fillStyle = "rgba(255, 255, 255, 0.6)";
				ctx.fill();
			}
		});

		ctx.restore();
	  };

	const drawCanvas = (poses, ctx, currentScore) => {
		ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

		const activePose = POSES[currentPoseName];
		const userPose = poses[0];

		// Change skeleton color based on score (Wiggle room logic)
		const isMatch = currentScore > 85;

		poses.forEach(({ keypoints }) => {
			// Draw the Colored Lines (Bones)
			SKELETON_CONNECTIONS.forEach((conn) => {
				const kp1 = keypoints[conn.pairs[0]];
				const kp2 = keypoints[conn.pairs[1]];

				if (kp1.score > 0.3 && kp2.score > 0.3) {
					ctx.beginPath();

					// If score > 80, make it green. Otherwise, keep original colors.
					ctx.strokeStyle = isMatch ? "#00FF00" : conn.color; // Flash Green on match
					ctx.lineWidth = isMatch ? 8 : 4; // Make it thicker when correct
					ctx.lineCap = "round";
					ctx.moveTo(kp1.x, kp1.y);
					ctx.lineTo(kp2.x, kp2.y);
					ctx.stroke();
				}
			});

			// Draw the Dots (Joints)
			keypoints.forEach((kp) => {
				if (kp.score > 0.3) {
					ctx.beginPath();
					ctx.arc(kp.x, kp.y, 4, 0, 2 * Math.PI);
					ctx.fillStyle = "white";
					ctx.fill();
					ctx.strokeStyle = "black";
					ctx.lineWidth = 1;
					ctx.stroke();
				}
			});
		});

		// Draw the guide anchored to the user
		drawAdaptiveGuide(ctx, userPose, { offsets: calculateOffsetsFromAngles(activePose.targetAngles) });

		// UI Overlay for Score
		ctx.fillStyle = "white";
		ctx.font = "30px Arial";
		ctx.fillText(`Match: ${currentScore}%`, 20, 50);


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

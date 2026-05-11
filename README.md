# Avatar Twister

**Avatar Twister** is a real-time, motion controlled web experience that merges physical movement with a virtual environment. By leveraging computer vision, the application mirrors a user's full body pose onto a digital avatar, challenging them to match specific poses under a ticking clock.

---

## Features

* **Full Body Motion Mirroring**: Real-time tracking that maps your physical movements directly onto a virtual avatar.
* **Pose Matching Engine**: An overlay system that displays target poses for the player to replicate.
* **Dynamic Scoring System**: Earn points by successfully holding poses before the timer expires.
* **Intuitive Calibration**: Start the game seamlessly by matching a standard **T-Pose** to calibrate the sensors.
* **Incremental Difficulty**: The game continues with new poses until the player fails to match one in time.

**Check out the Demo:** [https://ndbrewer20.github.io/Avatar-Tracking/](https://ndbrewer20.github.io/Avatar-Tracking/)

## Tech Stack

* **Frontend**: React.js
* **Motion Tracking**: TensorFlow.js
* **Rendering**: HTML5 Canvas / WebGL
* **Styling**: CSS

## How to Play

1. **Grant Camera Access**: Allow the browser to use your webcam for body tracking.
2. **Calibrate**: Stand back until your full body is visible. Match the **T-Pose** shown on screen to begin.
3. **Match the Overlay**: A target pose will appear. Move your body to align your avatar with the ghosted image.
4. **Beat the Clock**: Hold the pose until the timer hits zero to score points and move to the next round.
5. **Keep it Up**: The game ends when a pose is missed. Aim for a new high score!

## Installation & Setup

1. **Clone the repository**:
     ```bash
     git clone https://github.com/your-username/avatar-twister.git
     ```
2. **Install Dependencies**
    ```bash
    cd avatar-twister
    npm install
    ```
3. Run the Application
    ```bash
    npm start
    ```
4. Open ```http://localhost:3000``` in your browser

## Project Logic

The core of the project relies on calculating geometric joint angles and coordinate offsets. The scoring mechanism is gated by a visibility check ensuring that the camera has a clear view of the user's limbs before the challenge begins.

---
*Developed as an exploration of Pose Detection and Real-Time Web Interaction.*

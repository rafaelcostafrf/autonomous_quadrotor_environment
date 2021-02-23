# Quadrotor mathematical environment and 3D animation using python
# Main Branch - V6.2
## How to use the 3D environment:

	1. Download the whole repository
	2. Install the following packages:
		Panda3d 1.10.6.post2
		OpenCv 4.2.0
		Numpy 1.19.0
		Scipy 1.5.0
		PyTorch 1.5.1
	3. Run ./Main.py (does not work on Spyder console)
	4. Basic Controls:
		C - Changes camera
		WASD - Changes external camera angle
		RF - Changes external camera distance
	5. Controller:
		Machine Learning Based Controller, trained by a PPO algorithm. 
	6. Estimation Algorithm:
		MEMS - Simulates an onboard IMU, with gyroscope, accelerometer and magnetometer. 
			TRIAD algorithm is used to estimate attitude, retangular integrator is used to estimate position. 
		Hybrid - MEMS + Camera Estimation Algorithm
		True State - Uses the exact simulated state.

	

## How to use the mathematical environment:

	1. Download ./environment
	2. Install the following packages:
		Numpy 1.19.0
		Scipy 1.5.0
	3. Some examples in ./examples

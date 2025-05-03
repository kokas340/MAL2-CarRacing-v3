import numpy as np
import cv2  # OpenCV for image preprocessing
import neat

class NeatAgent:
    def __init__(self, genome, config):
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)

    def preprocess(self, obs):
        # obs is 96x96x3 RGB image
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (16, 12))  # downsample to 12×8 = 96 values
        normalized = resized.flatten() / 255.0  # scale to [0, 1]
        return normalized

    def act(self, obs):
        x = self.preprocess(obs)

        # Optional: add additional state features if needed
        # For now, using 3 dummy values: [velocity_x, velocity_y, angular_velocity]
        extras = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Combine image input with extras
        inputs = np.concatenate([x, extras])

        output = self.net.activate(inputs)

        # Map outputs to control range
        steer = np.clip(output[0], -1.0, 1.0)
        gas   = np.clip(output[1],  0.0, 1.0)
        brake = np.clip(output[2],  0.0, 1.0)

        return np.array([steer, gas, brake], dtype=np.float32)


 
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.0005, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.98   # Faster decay
        self.epsilon_min = 0.01     # Allow more exploitation

        # Training
        self.batch_size = 32
        self.memory = []
        self.max_memory = 5000
        self.model = self._build_model(lr)

        # Tracking
        self.loss_history = []

    def _build_model(self, lr):
        model = tf.keras.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='mse'
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(q_values[0])

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, targets = [], []

        for i in minibatch:
            state, action, reward, next_state, done = self.memory[i]
            target = self.model.predict(np.array([state]), verbose=0)[0]

            if done:
                target[action] = reward
            else:
                next_qs = self.model.predict(np.array([next_state]), verbose=0)[0]
                target[action] = reward + self.gamma * np.max(next_qs)

            states.append(state)
            targets.append(target)

        history = self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        self.loss_history.append(history.history['loss'][0])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        

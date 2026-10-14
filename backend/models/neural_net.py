import numpy as np

class MLP:
    """
    Vectorized 2-Layer Neural Network (MLP) with Adam Optimizer in NumPy.
    Input -> Dense(hidden_dim, ReLU) -> Dense(output_dim, linear/tanh)
    """
    def __init__(self, input_dim=5, hidden_dim=32, output_dim=3, lr=0.005, activation_out="linear"):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.activation_out = activation_out

        # He / Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

        # Adam optimizer state
        self.mW1, self.vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.mb1, self.vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.mW2, self.vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.mb2, self.vb2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        self.x = x
        self.z1 = np.dot(x, self.W1) + self.b1
        self.h1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = np.dot(self.h1, self.W2) + self.b2

        if self.activation_out == "tanh":
            self.out = np.tanh(self.z2)
        elif self.activation_out == "softmax":
            shift = self.z2 - np.max(self.z2, axis=-1, keepdims=True)
            ex = np.exp(shift)
            self.out = ex / np.sum(ex, axis=-1, keepdims=True)
        else:
            self.out = self.z2
        return self.out

    def backward(self, grad_output, return_grad_input=False):
        # grad_output shape: (batch_size, output_dim)
        if grad_output.ndim == 1:
            grad_output = grad_output.reshape(1, -1)

        batch_size = grad_output.shape[0]

        if self.activation_out == "tanh":
            dz2 = grad_output * (1.0 - self.out ** 2)
        elif self.activation_out == "softmax":
            dz2 = grad_output
        else:
            dz2 = grad_output

        dW2 = np.dot(self.h1.T, dz2) / batch_size
        db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size

        dh1 = np.dot(dz2, self.W2.T)
        dz1 = dh1 * (self.z1 > 0)  # ReLU grad

        dW1 = np.dot(self.x.T, dz1) / batch_size
        db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size

        # Adam update
        self.t += 1
        for p, g, m, v in [
            (self.W1, dW1, self.mW1, self.vW1),
            (self.b1, db1, self.mb1, self.vb1),
            (self.W2, dW2, self.mW2, self.vW2),
            (self.b2, db2, self.mb2, self.vb2)
        ]:
            m[:] = self.beta1 * m + (1 - self.beta1) * g
            v[:] = self.beta2 * v + (1 - self.beta2) * (g ** 2)
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        dx = np.dot(dz1, self.W1.T)
        return dx

    def copy_weights_from(self, source_net, tau=1.0):
        """Polyak soft update: target = tau * source + (1 - tau) * target"""
        self.W1 = tau * source_net.W1 + (1.0 - tau) * self.W1
        self.b1 = tau * source_net.b1 + (1.0 - tau) * self.b1
        self.W2 = tau * source_net.W2 + (1.0 - tau) * self.W2
        self.b2 = tau * source_net.b2 + (1.0 - tau) * self.b2

    def get_weights(self):
        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy()
        }

    def set_weights(self, weights):
        self.W1 = weights["W1"].copy()
        self.b1 = weights["b1"].copy()
        self.W2 = weights["W2"].copy()
        self.b2 = weights["b2"].copy()


class ReplayBuffer:
    """Cyclic Experience Replay Buffer for off-policy deep RL with terminal flag support."""
    def __init__(self, capacity=2000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state_feat, action, reward, next_state_feat, done=False):
        item = (state_feat, action, reward, next_state_feat, bool(done))
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in idxs:
            item = self.buffer[i]
            states.append(item[0])
            actions.append(item[1])
            rewards.append(item[2])
            next_states.append(item[3])
            dones.append(item[4] if len(item) > 4 else False)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_)
        )

    def __len__(self):
        return len(self.buffer)

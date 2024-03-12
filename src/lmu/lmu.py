import numpy as np
from scipy.signal import cont2discrete

from keras import activations
from keras import backend
from keras import ops
from keras import layers

from keras.src.layers.rnn.dropout_rnn_cell import DropoutRNNCell


class LMUCell(layers.Layer, DropoutRNNCell):
    """Cell class for the LMU layer.

    Reference:
        Voelker and Eliasmith (2018). Improving spiking dynamical
        networks: Accurate delays, higher-order synapses, and time cells.
        Neural Computation, 30(3): 569-609.

    Args:
        hidden_size: Size of the hidden vector.
        memory_size: Size of the memory vector (order of Legendre polynomials).
        theta: The number of timesteps in the sliding window that is represented
            using the LTI system.
        activation: Activation function to use. Default: hyperbolic tangent
            (`tanh`). If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        dropout: Float between 0 and 1. Fraction of the units to drop for the
            linear transformation of the inputs. Default: 0.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
            for the linear transformation of the recurrent state. Default: 0.
        seed: Random seed for dropout.

    Call arguments:
        inputs: A 2D tensor, with shape `(batch, features)`.
        states: A list of 2 2D tensor with shape `(batch, hidden_size)` and
            `(batch, memory_size)`, which is the state from the previous time step.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. Only relevant when `dropout` or
            `recurrent_dropout` is used.
    """

    def __init__(
        self,
        hidden_size,
        memory_size,
        theta,
        activation="tanh",
        dropout=0.0,
        recurrent_dropout=0.0,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.theta = theta
        self.activation = activations.get(activation)

        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
        self.seed = seed
        self.seed_generator = backend.random.SeedGenerator(seed=seed)

        self.state_size = [self.hidden_size, self.memory_size]
        self.output_size = self.hidden_size

        self.A, self.B = self._compute_state_space_matrices(memory_size, theta)

    def build(self, input_shape):
        super().build(input_shape)
        input_dim = input_shape[-1]

        # Kernels
        self.W_x = self.add_weight(
            shape=(self.hidden_size, input_dim),
            name="W_x",
            initializer="glorot_uniform",
        )
        self.W_h = self.add_weight(
            shape=(self.hidden_size, self.hidden_size),
            name="W_h",
            initializer="glorot_uniform",
        )
        self.W_m = self.add_weight(
            shape=(self.hidden_size, self.memory_size),
            name="W_m",
            initializer="glorot_uniform",
        )

        # Encoding vectors
        self.e_x = self.add_weight(
            shape=(1, input_dim),
            name="e_x",
            initializer=".lecun_uniform",
        )
        self.e_h = self.add_weight(
            shape=(1, self.hidden_size),
            name="e_h",
            initializer=".lecun_uniform",
        )
        self.e_m = self.add_weight(
            shape=(1, self.memory_size),
            name="e_m",
            initializer="zeros",
        )

        self.built = True

    def _compute_state_space_matrices(self, memory_size, theta):
        """Returns the discretized state space matrices A and B

        Taken from https://github.com/hrshtv/pytorch-lmu/blob/master/src/lmu.py
        """

        Q = np.arange(memory_size, dtype=np.float64).reshape(-1, 1)
        R = (2 * Q + 1) / theta
        i, j = np.meshgrid(Q, Q, indexing="ij")

        # Continuous
        A = R * np.where(i < j, -1, (-1.0) ** (i - j + 1))
        B = R * ((-1.0) ** Q)
        C = np.ones((1, memory_size))
        D = np.zeros((1,))

        # Convert to discrete
        A, B, C, D, dt = cont2discrete(system=(A, B, C, D), dt=1.0, method="zoh")

        return A, B

    def call(self, inputs, states, training=False):
        h_tm1 = states[0]  # previous hidden state
        m_tm1 = states[1]  # previous memory state

        dp_mask = self.get_dropout_mask(inputs)
        rec_dp_mask = self.get_recurrent_dropout_mask(h_tm1)

        if training and 0.0 < self.dropout < 1.0:
            inputs = inputs * dp_mask
        if training and 0.0 < self.recurrent_dropout < 1.0:
            h_tm1 = h_tm1 * rec_dp_mask

        # Equation (7) of the paper # [batch_size, 1]
        u = (
            ops.matmul(inputs, self.e_x)
            + ops.matmul(h_tm1, self.e_h)
            + ops.matmul(m_tm1, self.e_m)
        )

        # Equation (4) of the paper # [batch_size, memory_size]
        m = ops.matmul(m_tm1, self.A) + ops.matmul(u, self.B)

        # Equation (6) of the paper # [batch_size, hidden_size]
        h = self.activation(
            ops.matmul(inputs, self.W_x)
            + ops.matmul(h_tm1, self.W_h)
            + ops.matmul(m, self.W_m)
        )

        return h, [h, m]

    def get_config(self):
        config = {
            "hidden_size": self.hidden_size,
            "memory_size": self.memory_size,
            "theta": self.theta,
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def get_initial_state(self, batch_size=None):
        return [
            ops.zeros((batch_size, d), dtype=self.compute_dtype)
            for d in self.state_size
        ]


class LMU(layers.RNN):
    """Legendre Memory Unit layer.

    Reference:
        Voelker and Eliasmith (2018). Improving spiking dynamical
        networks: Accurate delays, higher-order synapses, and time cells.
        Neural Computation, 30(3): 569-609.

    Args:
        hidden_size: Size of the hidden vector.
        memory_size: Size of the memory vector (order of Legendre polynomials).
        theta: The number of timesteps in the sliding window that is represented
            using the LTI system.
        activation: Activation function to use. Default: hyperbolic tangent
            (`tanh`). If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        dropout: Float between 0 and 1. Fraction of the units to drop for the
            linear transformation of the inputs. Default: 0.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
            for the linear transformation of the recurrent state. Default: 0.
        seed: Random seed for dropout.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence. Default: `False`.
        return_state: Boolean. Whether to return the last state in addition
            to the output. Default: `False`.
        go_backwards: Boolean (default: `False`).
            If `True`, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default: `False`). If `True`, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If `True`, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.

    Call arguments:
        inputs: A 3D tensor, with shape `(batch, timesteps, feature)`.
        mask: Binary tensor of shape `(samples, timesteps)` indicating whether
            a given timestep should be masked  (optional).
            An individual `True` entry indicates that the corresponding timestep
            should be utilized, while a `False` entry indicates that the
            corresponding timestep should be ignored. Defaults to `None`.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. This argument is passed to the
            cell when calling it. This is only relevant if `dropout` or
            `recurrent_dropout` is used  (optional). Defaults to `None`.
        initial_state: List of initial state tensors to be passed to the first
            call of the cell (optional, `None` causes creation
            of zero-filled initial state tensors). Defaults to `None`.
    """

    def __init__(
        self,
        hidden_size,
        memory_size,
        theta,
        activation="tanh",
        dropout=0.0,
        recurrent_dropout=0.0,
        seed=None,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=False,
        activity_regularizer=None,
        **kwargs,
    ):
        cell = LMUCell(
            hidden_size=hidden_size,
            memory_size=memory_size,
            theta=theta,
            activation=activation,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            seed=seed,
        )
        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            activity_regularizer=activity_regularizer,
            **kwargs,
        )
        self.input_spec = layers.InputSpec(ndim=3)

    @property
    def hidden_size(self):
        return self.cell.hidden_size

    @property
    def memory_size(self):
        return self.cell.memory_size

    @property
    def theta(self):
        return self.cell.theta

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {
            "hidden_size": self.hidden_size,
            "memory_size": self.memory_size,
            "theta": self.theta,
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "seed": self.cell.seed,
        }
        base_config = super().get_config()
        del base_config["cell"]
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

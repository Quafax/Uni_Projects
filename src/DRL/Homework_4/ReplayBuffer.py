import numpy as np
import tensorflow as tf

#just the exact same as in the homeowkr pdf
class ReplayBuffer:
    def __init__(self, capacity, obs_dim):
        self.capacity, self.obs_dim = capacity, obs_dim
        self.s = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros(capacity, np.int32)
        self.r = np.zeros(capacity, np.float32)
        self.terminal = np.zeros(capacity, bool)

        # POINTER TO THE INDEX OF NEXT ENTRY
        self.ptr = 0
        self.n = 0

    def add(self, s, a=0, r=0.0, terminal=False):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.terminal[i] = (s, a, r, terminal)

        self.ptr = (i + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def sample(self):
        newest = (self.ptr - 1) % self.capacity
        i = np.random.randint(self.n)

        while i == newest or self.terminal[i]:
            i = np.random.randint(self.n)

        j = (i + 1) % self.capacity

        return (
            self.s[i],
            self.a[i],
            self.r[i],
            self.s[j],
            np.float32(self.terminal[j])
        )

    def as_dataset(self, batch_size):
        def gen():
            while True:
                yield self.sample()

        signature = (
            tf.TensorSpec((self.obs_dim,), tf.float32),
            tf.TensorSpec((), tf.int32),
            tf.TensorSpec((), tf.float32),
            tf.TensorSpec((self.obs_dim,), tf.float32),
            tf.TensorSpec((), tf.float32)
        )

        dataset = tf.data.Dataset.from_generator(
            gen, output_signature=signature
        )

        process = lambda ds: ds.batch(batch_size).prefetch(
            tf.data.AUTOTUNE
        )

        dataset = dataset.apply(process)

        return iter(dataset)

    def save(self, path):
        np.savez(
            path,
            s=self.s,
            a=self.a,
            r=self.r,
            terminal=self.terminal,
            ptr=self.ptr,
            n=self.n
        )

    def load(self, path):
        if not path.endswith(".npz"):
            path += ".npz"

        d = np.load(path)

        self.s = d["s"]
        self.a = d["a"]
        self.r = d["r"]
        self.terminal = d["terminal"]

        self.ptr, self.n = int(d["ptr"]), int(d["n"])
        self.capacity, self.obs_dim = self.s.shape
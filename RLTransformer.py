import tensorflow as tf
from tensorflow.keras import layers
from einops import rearrange


### RMSNorm ###
class RMSNorm(layers.Layer):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by default
                     because RMSNorm doesn't enforce re-centering invariance.
        """
        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        stdv = 1. / tf.math.sqrt(d / 3)
        uniform_init = tf.random_uniform_initializer(minval=-stdv, maxval=stdv)
        zeros_init = tf.zeros_initializer()

        self.scale = tf.Variable(
            initial_value=uniform_init(shape=(d,), dtype="float32"), trainable=True, name="scale"
        )
        if self.bias:
            self.offset = tf.Variable(
                initial_value=zeros_init(shape=(d,), dtype="float32"), trainable=True, name="offset"
            )

    def call(self, inputs, **kwargs):
        if len(inputs.shape) >= 2:
            z = int(len(inputs.shape) - 1)
            axis = (0 * z, 1)  # matrix
        else:
            axis = None  # vector

        if self.p < 0. or self.p > 1.:
            norm_x = tf.norm(inputs, ord="fro", axis=axis, keepdims=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = tf.split(inputs, [partial_size, self.d - partial_size], axis=-1)
            norm_x = tf.norm(partial_x, ord="fro", axis=axis, keepdims=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = inputs / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


### FFNN ###
class FeedForward(layers.Layer):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.ffnn = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(dim, activation=None),
            layers.Dropout(dropout)
        ])

    def call(self, inputs, **kwargs):
        return self.ffnn(inputs)


### PreNorm ###
class PreNorm(layers.Layer):
    def __init__(self, fn):
        super(PreNorm, self).__init__()
        self.norm = layers.LayerNormalization(axis=-1, epsilon=1e-5)
        self.fn = fn

    def call(self, inputs, **kwargs):
        return self.fn(self.norm(inputs), **kwargs)


### Attention ###
class RectifiedLinearAttention(layers.Layer):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., rmsnorm=False):
        super(RectifiedLinearAttention, self).__init__()
        innder_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = layers.Dense(innder_dim * 3, use_bias=False)

        self.norm = RMSNorm(innder_dim) if rmsnorm else layers.LayerNormalization(axis=-1, epsilon=1e-5)

        self.to_out = tf.keras.Sequential([
            layers.Dense(dim),
            layers.Dropout(dropout)
        ]) if project_out else tf.identity

    def call(self, inputs, **kwargs):
        b, n, _, h = *inputs.shape, self.heads
        qkv = tf.split(self.to_qkv(inputs), num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = tf.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = tf.keras.activations.relu(dots)

        out = tf.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(self.norm(out))
        return out


### Transformer Encoder ###
class RLTransformer(layers.Layer):
    def __init__(self, depth, dim, heads, dim_head, scale, dropout):
        super(RLTransformer, self).__init__()
        self.layers = []
        for _ in range(depth):
            self.layers.append([
                PreNorm(RectifiedLinearAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout, rmsnorm=True)),
                PreNorm(FeedForward(dim, dim * scale, dropout=dropout))
            ])

    def call(self, inputs, **kwargs):
        for attn, ff in self.layers:
            inputs = attn(inputs) + inputs
            inputs = ff(inputs) + inputs
        return inputs

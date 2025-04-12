const tf = require('@tensorflow/tfjs');

class MultiHeadAttention extends tf.layers.Layer {
    constructor(numHeads, embedDim) {
        super({});
        this.numHeads = numHeads;
        this.embedDim = embedDim;
        if (embedDim % numHeads !== 0) {
            throw new Error('Embedding dimension must be divisible by number of heads');
        }
        this.projectionDim = embedDim / numHeads;
    }

    build(inputShape) {
        this.queryDense = this.addWeight(
            'queryDense',
            [this.embedDim, this.embedDim],
            'float32',
            tf.initializers.glorotUniform()
        );
        this.keyDense = this.addWeight(
            'keyDense',
            [this.embedDim, this.embedDim],
            'float32',
            tf.initializers.glorotUniform()
        );
        this.valueDense = this.addWeight(
            'valueDense',
            [this.embedDim, this.embedDim],
            'float32',
            tf.initializers.glorotUniform()
        );
        this.combineHeadsDense = this.addWeight(
            'combineHeadsDense',
            [this.embedDim, this.embedDim],
            'float32',
            tf.initializers.glorotUniform()
        );
        super.build(inputShape);
    }

    splitHeads(x, batchSize) {
        const reshaped = x.reshape([batchSize, -1, this.numHeads, this.projectionDim]);
        return reshaped.transpose([0, 2, 1, 3]); // [batch, heads, seq, projection_dim]
    }

    scaledDotProductAttention(q, k, v) {
        const matmulQK = tf.matMul(q, k, false, true); // [batch, heads, seq_q, seq_k]
        const dk = tf.scalar(this.projectionDim, 'float32');
        const scaledAttentionLogits = matmulQK.div(tf.sqrt(dk));
        const attentionWeights = tf.softmax(scaledAttentionLogits, -1); // softmax on last axis
        const output = tf.matMul(attentionWeights, v); // [batch, heads, seq_q, projection_dim]
        return output;
    }

    call(inputs, kwargs) {
        const [query, key, value] = inputs;
        const batchSize = query.shape[0];

        // 使用 tf.matMul 替代 tf.dot
        let q = tf.matMul(query, this.queryDense.read());
        let k = tf.matMul(key, this.keyDense.read());
        let v = tf.matMul(value, this.valueDense.read());

        q = this.splitHeads(q, batchSize);
        k = this.splitHeads(k, batchSize);
        v = this.splitHeads(v, batchSize);

        const attention = this.scaledDotProductAttention(q, k, v);

        const attentionTransposed = attention.transpose([0, 2, 1, 3]);
        const concatAttention = attentionTransposed.reshape([batchSize, -1, this.embedDim]);

        const output = tf.matMul(concatAttention, this.combineHeadsDense.read());

        return output;
    }

    computeOutputShape(inputShape) {
        const shape = inputShape[0]; // query shape
        return [shape[0], shape[1], this.embedDim];
    }

    getClassName() {
        return 'MultiHeadAttention';
    }
}

module.exports = { MultiHeadAttention };

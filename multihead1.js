import * as tf from '@tensorflow/tfjs';

export class MultiHeadAttentionLayer extends tf.layers.Layer {
    constructor(numHeads, keyDim, config) {
        super(config || {});
        this.numHeads = numHeads;
        this.keyDim = keyDim;
        this.denseQ = tf.layers.dense({ units: numHeads * keyDim });
        this.denseK = tf.layers.dense({ units: numHeads * keyDim });
        this.denseV = tf.layers.dense({ units: numHeads * keyDim });
    }

    computeAttention(query, key, value) {
        const scores = tf.matMul(query, key.transpose([0, 2, 1]))
            .div(tf.sqrt(tf.scalar(this.keyDim)));
        const weights = tf.softmax(scores, -1);
        return tf.matMul(weights, value);
    }

    call(inputs) {
        const [q, k, v] = inputs;
        const batchSize = q.shape[0];

        const qProj = this.denseQ.apply(q).reshape([batchSize, -1, this.numHeads, this.keyDim]);
        const kProj = this.denseK.apply(k).reshape([batchSize, -1, this.numHeads, this.keyDim]);
        const vProj = this.denseV.apply(v).reshape([batchSize, -1, this.numHeads, this.keyDim]);

        const heads = [];
        for (let i = 0; i < this.numHeads; i++) {
            const head = this.computeAttention(
                qProj.slice([0, 0, i, 0], [batchSize, -1, 1, -1]).squeeze([2]),
                kProj.slice([0, 0, i, 0], [batchSize, -1, 1, -1]).squeeze([2]),
                vProj.slice([0, 0, i, 0], [batchSize, -1, 1, -1]).squeeze([2])
            );
            heads.push(head);
        }

        return tf.concat(heads, -1);
    }

    static get className() {
        return 'MultiHeadAttentionLayer';
    }
}

tf.serialization.registerClass(MultiHeadAttentionLayer);

// 定义输入（假设输入形状为[序列长度, 特征维度]）
// test();

function test() {
    const qInput = tf.input({ shape: [10, 4] });
 

    // 实例化多头注意力层
    const mhaLayer = new MultiHeadAttentionLayer(8, 32); // 8个头，每个key维度32


    // 应用注意力层
    const mhaOutput = mhaLayer.apply([qInput, qInput, qInput]);

    // 构建模型
    const model = tf.model({
        inputs: [qInput, kInput, vInput],
        outputs: mhaOutput
    });

    // 打印模型结构
    model.summary();
}




// module.exports = { MultiHeadAttentionLayer }
import * as tf from '@tensorflow/tfjs';
import type { Tensor, Shape } from '@tensorflow/tfjs';
import type { Dense } from "@tensorflow/tfjs-layers/dist/layers/core"

export class MultiHeadAttentionLayer extends tf.layers.Layer {
    numHeads: number;
    keyDim: number;
    denseQ: Dense;
    denseK: Dense;
    denseV: Dense;
    constructor(numHeads: number, keyDim:number, config) {
        super(config || {});
        this.numHeads = numHeads;
        this.keyDim = keyDim;
        this.denseQ = tf.layers.dense({ units: numHeads * keyDim });
        this.denseK = tf.layers.dense({ units: numHeads * keyDim });
        this.denseV = tf.layers.dense({ units: numHeads * keyDim });
    }

    computeAttention(query: tf.Tensor , key: tf.Tensor, value: tf.Tensor): Tensor {
        const scores = tf.matMul(query, key.transpose([0, 2, 1]))
            .div(tf.sqrt(tf.scalar(this.keyDim)));
        const weights = tf.softmax(scores, -1);
        return tf.matMul(weights, value);
    }

    call(inputs) {
        const [q, k, v] = inputs;
        const batchSize = q.shape[0];

        const qProj =  tf.reshape( this.denseQ.apply(q), [batchSize, -1, this.numHeads, this.keyDim]);
        const kProj = tf.reshape( this.denseK.apply(k), [batchSize, -1, this.numHeads, this.keyDim]);
        const vProj = tf.reshape( this.denseV.apply(v), [batchSize, -1, this.numHeads, this.keyDim]);

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

    callAndReturnAttentionScores(
        query: Tensor
      ): [Tensor, Tensor] {
        return tidy(() => {
          if (!this.builtFromSignature) {
            this.buildFromSignature(
              query.shape,
              value.shape,
              key ? key.shape : null
            );
          }
          if (key == null) {
            key = value;
          }
    
          // TODO(pforderique): Support RaggedTensor inputs.
    
          attentionMask = this.computeAttentionMask(
            query,
            value,
            attentionMask,
            useCausalMask,
          );
    
          //   N = `numAttentionHeads`
          //   H = `sizePerHead`
          // `query` = [B, T, N ,H]
          query = this.queryDense.apply(query) as Tensor;
    
          // `key` = [B, S, N, H]
          key = this.keyDense.apply(key) as Tensor;
    
          // `value` = [B, S, N, H]
          value = this.valueDense.apply(value) as Tensor;
    
          const [attentionOutputPreDense, attentionScores] = this.computeAttention(
            query,
            key,
            value,
            attentionMask,
            training
          );
          const attentionOutput =
            this.outputDense.apply(attentionOutputPreDense) as Tensor;
    
          return [attentionOutput, attentionScores];
        });
    }
    override computeOutputShape(inputShapes: [Shape, Shape, Shape]): Shape |Shape[]{
        const [queryShape, valueShape, maybeKeyShape] = inputShapes;
        const keyShape = maybeKeyShape ?? valueShape;
    
        if (queryShape.slice(-1)[0] !== valueShape.slice(-1)[0]) {
          throw new ValueError(
            `The last dimension of 'queryShape' and 'valueShape' must be equal, ` +
            `but are ${queryShape.slice(-1)[0]}, ${valueShape.slice(-1)[0]}. ` +
            `Received: queryShape=${queryShape}, valueShape=${valueShape}`
          );
        }
    
        if (!util.arraysEqual(valueShape.slice(1, -1), keyShape.slice(1, -1))) {
          throw new Error(
            `All dimensions of 'value' and 'key', except the last one, must be ` +
            `equal. Received ${valueShape} and ${keyShape}`
          );
        }
    
        if (this._outputShape) {
          return queryShape.slice(0, -1).concat(this._outputShape);
        }
    
        return queryShape;
      }
    
    static get className() {
        return 'MultiHeadAttentionLayer';
    }
}

tf.serialization.registerClass(MultiHeadAttentionLayer);

// 定义输入（假设输入形状为[序列长度, 特征维度]）
test();

function test() {
    const qInput = tf.input({ shape: [10, 4] });
    const kInput = tf.input({ shape: [10, 4] });
    const vInput = tf.input({ shape: [10, 4] });

    // 实例化多头注意力层
    const mhaLayer = new MultiHeadAttentionLayer(8, 32); // 8个头，每个key维度32


    // 应用注意力层
    const mhaOutput = mhaLayer.apply([qInput, kInput, vInput]);

    // 构建模型
    const model = tf.model({
        inputs: [qInput, kInput, vInput],
        outputs: mhaOutput
    });

    model.fit()

    // 打印模型结构
    model.summary();
}




// module.exports = { MultiHeadAttentionLayer }
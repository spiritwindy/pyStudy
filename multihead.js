import  tf from '@tensorflow/tfjs';

export class MultiHeadAttention {
    constructor(numHeads, keyDim) {
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
  
    apply(inputs) {
      const [q, k, v] = inputs;
      const batchSize = q.shape[0];
  
      // 线性变换
      const qProj = this.denseQ.apply(q).reshape([batchSize, -1, this.numHeads, this.keyDim]);
      const kProj = this.denseK.apply(k).reshape([batchSize, -1, this.numHeads, this.keyDim]);
      const vProj = this.denseV.apply(v).reshape([batchSize, -1, this.numHeads, this.keyDim]);
  
      // 分头计算
      const heads = [];
      for (let i = 0; i < this.numHeads; i++) {
        const head = this.computeAttention(
          qProj.slice([0, 0, i, 0], [batchSize, -1, 1, -1]).squeeze([2]),
          kProj.slice([0, 0, i, 0], [batchSize, -1, 1, -1]).squeeze([2]),
          vProj.slice([0, 0, i, 0], [batchSize, -1, 1, -1]).squeeze([2])
        );
        heads.push(head);
      }
  
      // 合并输出
      return tf.concat(heads, -1);
    }
  }
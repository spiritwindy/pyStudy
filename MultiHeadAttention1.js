const tf = require('@tensorflow/tfjs');

class HeadSliceLayer extends tf.layers.Layer {
    constructor(i) {
      super({});
      this.i = i;
    }
  
    call(inputs) {
      return inputs.slice([0, 0, this.i, 0], [-1, -1, 1, -1]).squeeze([2]);
    }
  
    static get className() {
      return 'HeadSliceLayer';
    }
  }
  

class MultiHeadAttention {
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

        // 1. 先做点积：batch-wise matmul
  
  // 2. 除以 sqrt(keyDim)
  score = tf.layers.multiply().apply([
    score,
    tf.scalar(1 / Math.sqrt(this.keyDim))
  ]);
      const weights = tf.softmax(scores, -1);
      return tf.matMul(weights, value);
    }
  
    apply(inputs) {
      const [q, k, v] = inputs;
      const batchSize = q.shape[0];
  
      // 线性变换
      // const qProj = this.denseQ.apply(q).reshape([batchSize, -1, this.numHeads, this.keyDim]);
      // const kProj = this.denseK.apply(k).reshape([batchSize, -1, this.numHeads, this.keyDim]);
      // const vProj = this.denseV.apply(v).reshape([batchSize, -1, this.numHeads, this.keyDim]);
      const reshape = tf.layers.reshape({targetShape: [batchSize, -1, this.numHeads, this.keyDim]});
      const qProj = reshape.apply(this.denseQ.apply(q));
      const kProj = reshape.apply(this.denseK.apply(k));
      const vProj = reshape.apply(this.denseV.apply(v));
      // 分头计算
      const heads = [];


      for (let i = 0; i < this.numHeads; i++) {
        const head = this.computeAttention(
            new HeadSliceLayer(i).apply(qProj), // qProj.slice([0, 0, i, 0], [batchSize, -1, 1, -1]).squeeze([2]),
            new HeadSliceLayer(i).apply(kProj),
          new HeadSliceLayer(i).apply(vProj)
        );
        heads.push(head);
      }
  
      // 合并输出
      return tf.concat(heads, -1);
    }
  }
    
module.exports = { MultiHeadAttention };
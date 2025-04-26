import * as tf from '@tensorflow/tfjs';

// 辅助函数：生成位置编码
const getPositionalEncoding = (maxLen, dModel) => {
  const position = tf.range(0, maxLen, 1, 'float32').expandDims(1);
  const divTerm = tf.exp(tf.range(0, dModel, 2, 'float32')
    .mul(-Math.log(10000.0) / dModel));
  
  const pe = tf.zeros([maxLen, dModel]);
  const sinValues = tf.sin(position.mul(divTerm));
  const cosValues = tf.cos(position.mul(divTerm));
  
  // 交替填充sin和cos值
  const peEven = pe.slice([0, 0], [-1, 1]).add(sinValues);
  const peOdd = pe.slice([0, 1], [-1, 1]).add(cosValues);
  return peEven.concat(peOdd, 1).reshape([maxLen, dModel]);
};

export class MultiHeadAttention {
  constructor(numHeads, keyDim, maxLen=512) {
    this.numHeads = numHeads;
    this.keyDim = keyDim;
    this.dModel = numHeads * keyDim;  // 输入维度需要与位置编码匹配
    this.maxLen = maxLen;
    
    // 初始化可训练层
    this.denseQ = tf.layers.dense({ units: this.dModel });
    this.denseK = tf.layers.dense({ units: this.dModel });
    this.denseV = tf.layers.dense({ units: this.dModel });
    
    // 预先生成位置编码矩阵
    this.positionalEncoding = getPositionalEncoding(maxLen, this.dModel);
  }

  // 添加位置编码到输入
  addPositionEncoding(input) {
    const seqLen = input.shape[1];
    const pe = this.positionalEncoding.slice([0, 0], [seqLen, -1]);
    return input.add(pe);
  }

  computeAttention(query, key, value) {
    const scores = tf.matMul(query, key.transpose([0, 2, 1]))
      .div(tf.sqrt(tf.scalar(this.keyDim)));
    const weights = tf.softmax(scores, -1);
    return tf.matMul(weights, value);
  }

  apply(inputs) {
    const [q, k, v] = inputs.map(input => this.addPositionEncoding(input));
    const batchSize = q.shape[0];

    // 线性变换 + 分头
    const project = (layer, x) => layer.apply(x).reshape([batchSize, -1, this.numHeads, this.keyDim]);
    const qProj = project(this.denseQ, q);
    const kProj = project(this.denseK, k);
    const vProj = project(this.denseV, v);

    // 并行计算多个注意力头
    const heads = Array(this.numHeads).fill().map((_, i) => {
      const extractHead = t => t.slice([0, 0, i, 0], [batchSize, -1, 1, -1]).squeeze([2]);
      return this.computeAttention(extractHead(qProj), extractHead(kProj), extractHead(vProj));
    });

    return tf.concat(heads, -1);
  }
}
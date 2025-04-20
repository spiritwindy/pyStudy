// 多头注意力层 MultiHeadAttention 实现示例
// 继承自 tf.layers.Layer，支持自定义头数、key/value 维度和可选遮罩
import * as tf from '@tensorflow/tfjs';
export class MultiHeadAttention extends tf.layers.Layer {
  /**
   * @param {Object} config
   * @param {number} config.numHeads - 注意力头数
   * @param {number} config.keyDim - 每个头的 key/query 维度
   * @param {number} [config.valueDim] - 每个头的 value 维度 (默认与 keyDim 相同)
   * @param {boolean} [config.useBias] - 是否在投影层使用 bias
   * @param {string|tf.initializer.Initializer} [config.kernelInitializer] - 权重初始化方法或初始化器实例
   */
  constructor(config) {
    super(config);
    this.numHeads = config.numHeads;
    this.keyDim = config.keyDim;
    this.valueDim = config.valueDim || config.keyDim;
    this.useBias = config.useBias != null ? config.useBias : true;
    this.kernelInitializer = config.kernelInitializer || 'glorotUniform';
    this.supportsMasking = true;
  }

  build(inputShape) {
    // 处理多输入或单输入情况
    let queryShape, keyShape, valueShape;
    if (Array.isArray(inputShape)) {
      queryShape = inputShape[0];
      keyShape = inputShape[1] || queryShape;
      valueShape = inputShape[2] || keyShape;
    } else {
      queryShape = keyShape = valueShape = inputShape;
    }
    if (!queryShape || queryShape.length < 3) {
      throw new Error('Input shape must be at least rank 3 [batchSize, seqLen, dim]');
    }
    const dim = queryShape[2];

    // 确保初始化器是 Initializer 对象
    const kernelInit =
      typeof this.kernelInitializer === 'string'
        ? tf.initializers[this.kernelInitializer]()
        : this.kernelInitializer;
    const biasInit = tf.initializers.zeros();

    // Q, K, V 的全连接投影权重
    this.qKernel = this.addWeight(
      'qKernel', [dim, this.numHeads * this.keyDim], null, kernelInit);
    this.kKernel = this.addWeight(
      'kKernel', [dim, this.numHeads * this.keyDim], null, kernelInit);
    this.vKernel = this.addWeight(
      'vKernel', [dim, this.numHeads * this.valueDim], null, kernelInit);
    this.oKernel = this.addWeight(
      'oKernel', [this.numHeads * this.valueDim, dim], null, kernelInit);

    if (this.useBias) {
      this.qBias = this.addWeight(
        'qBias', [this.numHeads * this.keyDim], null, biasInit);
      this.kBias = this.addWeight(
        'kBias', [this.numHeads * this.keyDim], null, biasInit);
      this.vBias = this.addWeight(
        'vBias', [this.numHeads * this.valueDim], null, biasInit);
      this.oBias = this.addWeight('oBias', [dim], null, biasInit);
    }

    this.built = true;
  }

  computeOutputShape(inputShape) {
    // 输出形状与 query 相同: [batchSize, seqLenQ, dim]
    let queryShape = Array.isArray(inputShape) ? inputShape[0] : inputShape;
    return queryShape;
  }

  /**
   * @param {Tensor|Tensor[]} inputs - [query, key, value] 或仅 [query] 或 Tensor
   * @param {Object} kwargs
   * @param {Tensor} [kwargs.mask] - 可选 mask，形状 [batch, seqQ, seqK]
   */
  call(inputs, kwargs) {
    let q, k, v;
    if (Array.isArray(inputs)) {
      q = inputs[0];
      k = inputs[1] || q;
      v = inputs[2] || q;
    } else {
      q = k = v = inputs;
    }
    const mask = kwargs && kwargs.mask;

    // 线性投影
    q = tf.matMul(q, this.qKernel.read());
    k = tf.matMul(k, this.kKernel.read());
    v = tf.matMul(v, this.vKernel.read());
    if (this.useBias) {
      q = tf.add(q, this.qBias.read());
      k = tf.add(k, this.kBias.read());
      v = tf.add(v, this.vBias.read());
    }

    // 获取尺寸信息
    const batchSize = q.shape[0];
    const seqLenQ = q.shape[1];
    const seqLenK = k.shape[1];
    const headDimQ = this.keyDim;
    const headDimV = this.valueDim;

    // 重塑并转置以分头
    q = tf.transpose(
      tf.reshape(q, [batchSize, seqLenQ, this.numHeads, headDimQ]),
      [0, 2, 1, 3]
    );
    k = tf.transpose(
      tf.reshape(k, [batchSize, seqLenK, this.numHeads, headDimQ]),
      [0, 2, 1, 3]
    );
    v = tf.transpose(
      tf.reshape(v, [batchSize, seqLenK, this.numHeads, headDimV]),
      [0, 2, 1, 3]
    );

    // 缩放点积注意力
    let scores = tf.matMul(q, k, false, true);
    const scale = Math.sqrt(headDimQ);
    scores = tf.div(scores, tf.scalar(scale));
    if (mask) {
      const maskTensor = tf.cast(mask, 'float32');
      const addMask = tf.mul(tf.sub(tf.scalar(1.0), maskTensor), tf.scalar(-1e9));
      scores = tf.add(scores, addMask);
    }
    const attn = tf.softmax(scores, -1);

    // 加权求和
    let context = tf.matMul(attn, v);
    context = tf.transpose(context, [0, 2, 1, 3]);
    const concatDim = this.numHeads * headDimV;
    context = tf.reshape(context, [batchSize, seqLenQ, concatDim]);

    // 输出投影
    let output = tf.matMul(context, this.oKernel.read());
    if (this.useBias) {
      output = tf.add(output, this.oBias.read());
    }
    return output;
  }

  getConfig() {
    const baseConfig = super.getConfig();
    return Object.assign({}, baseConfig, {
      numHeads: this.numHeads,
      keyDim: this.keyDim,
      valueDim: this.valueDim,
      useBias: this.useBias,
      kernelInitializer: this.kernelInitializer
    });
  }

  static get className() {
    return 'MultiHeadAttention';
  }
}
/**
 * 
 * @param {number} dModel 
 * @param {number} numHeads 
 * @returns {tf.Model}
 */
function transformerBlock(dModel, numHeads) {
  const input = tf.input({ shape: [null, dModel] }); // 输入序列

  // 多头注意力（Q=K=V=输入）
  const attention = new MultiHeadAttention({ numHeads, keyDim: dModel / numHeads })
    .apply([input, input, input]); // 自注意力机制

  // 残差连接 + 层归一化
  const add1 = tf.layers.add().apply([input, attention]);
  console.log("add1", add1.shape);
  const norm1 = tf.layers.layerNormalization().apply(add1);
  console.log("norm1",norm1.shape)
  // 前馈网络
  const dense1 = tf.layers.dense({ units: 4 * dModel, activation: 'relu' }).apply(norm1);
  console.log("dense1",dense1.shape)
  const dense2 = tf.layers.dense({ units: dModel }).apply(dense1);
  console.log("dense2",dense2.shape)
  // 残差连接 + 层归一化
  const add2 = tf.layers.add().apply([norm1, dense2]);
  const output = tf.layers.layerNormalization().apply(add2);
  console.log("output",output.shape)
  let model = tf.model({ inputs: input, outputs: output });
  model.summary()
  return model;
}

// 注册类以支持序列化和模型保存/加载
tf.serialization.registerClass(MultiHeadAttention);




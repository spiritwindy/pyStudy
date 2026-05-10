// 多头注意力层 MultiHeadAttention 实现示例
// 继承自 tf.layers.Layer，支持自定义头数、key/value 维度和可选遮罩
import * as tf from '@tensorflow/tfjs';

const MIN_INPUT_RANK = 3;
const BATCH_AXIS = 0;
const SEQUENCE_AXIS = 1;
const LAST_AXIS = -1;
const INFER_DIM = -1;
const MASK_FILL_VALUE = -1e9;
const SWAP_HEAD_AND_SEQUENCE_AXES = [0, 2, 1, 3];

export class MultiHeadAttention extends tf.layers.Layer {
  normalizeInputShapes(inputShape) {
    if (!Array.isArray(inputShape)) {
      return [inputShape, inputShape, inputShape];
    }
    const [queryShape, keyShape = queryShape, valueShape = keyShape] = inputShape;
    return [queryShape, keyShape, valueShape];
  }

  normalizeInputs(inputs) {
    if (!Array.isArray(inputs)) {
      return [inputs, inputs, inputs];
    }
    const [query, key = query, value = key] = inputs;
    return [query, key, value];
  }

  getFeatureDim(shape) {
    const featureDim = shape[shape.length + LAST_AXIS];
    if (featureDim == null) {
      throw new Error('The last input dimension must be defined.');
    }
    return featureDim;
  }

  getProjectionDim(headDim) {
    return this.numHeads * headDim;
  }

  projectInput(input, kernel, bias, outputDim) {
    const inputShape = input.shape;
    const featureDim = this.getFeatureDim(inputShape);
    const flatInput = tf.reshape(input, [INFER_DIM, featureDim]);
    let projected = tf.matMul(flatInput, kernel.read());
    if (this.useBias && bias) {
      projected = tf.add(projected, bias.read());
    }
    return tf.reshape(projected, inputShape.slice(0, LAST_AXIS).concat([outputDim]));
  }

  splitHeads(input, headDim) {
    const batchSize = input.shape[BATCH_AXIS];
    const sequenceLength = input.shape[SEQUENCE_AXIS];
    const reshaped = tf.reshape(input, [batchSize, sequenceLength, this.numHeads, headDim]);
    return tf.transpose(reshaped, SWAP_HEAD_AND_SEQUENCE_AXES);
  }

  combineHeads(input, sequenceLength, headDim) {
    const batchSize = input.shape[BATCH_AXIS];
    const combinedDim = this.getProjectionDim(headDim);
    const transposed = tf.transpose(input, SWAP_HEAD_AND_SEQUENCE_AXES);
    return tf.reshape(transposed, [batchSize, sequenceLength, combinedDim]);
  }

  /**
   * @param {Object} config
   * @param {number} config.numHeads - 注意力头数
   * @param {number} config.keyDim - 每个头的 key/query 维度
   * @param {number} [config.valueDim] - 每个头的 value 维度 (默认与 keyDim 相同)
   * @param {boolean} [config.useBias] - 是否在投影层使用 bias
   * @param {string|tf.initializer.Initializer} [config.kernelInitializer] - 权重初始化方法或初始化器实例
   */
  constructor(config) {
    const layerConfig = config && typeof config === 'object' ? config : {};
    super(layerConfig);
    if (!config || typeof config !== 'object') {
      throw new Error('MultiHeadAttention config must be an object, e.g. { numHeads, keyDim }.');
    }
    if (!Number.isInteger(config.numHeads) || config.numHeads <= 0) {
      throw new Error('MultiHeadAttention numHeads must be a positive integer.');
    }
    if (!Number.isInteger(config.keyDim) || config.keyDim <= 0) {
      throw new Error('MultiHeadAttention keyDim must be a positive integer.');
    }
    if (config.valueDim != null && (!Number.isInteger(config.valueDim) || config.valueDim <= 0)) {
      throw new Error('MultiHeadAttention valueDim must be a positive integer when provided.');
    }
    this.numHeads = config.numHeads;
    this.keyDim = config.keyDim;
    this.valueDim = config.valueDim || config.keyDim;
    this.useBias = config.useBias != null ? config.useBias : true;
    this.kernelInitializer = config.kernelInitializer || 'glorotUniform';
    this.supportsMasking = true;
  }

  build(inputShape) {
    // 处理多输入或单输入情况
    const [queryShape, keyShape, valueShape] = this.normalizeInputShapes(inputShape);
    if (!queryShape || queryShape.length < MIN_INPUT_RANK) {
      throw new Error('Input shape must be at least rank 3 [batchSize, seqLen, dim]');
    }
    const queryDim = this.getFeatureDim(queryShape);
    const keyInputDim = this.getFeatureDim(keyShape);
    const valueInputDim = this.getFeatureDim(valueShape);
    const keyProjectionDim = this.getProjectionDim(this.keyDim);
    const valueProjectionDim = this.getProjectionDim(this.valueDim);

    // 确保初始化器是 Initializer 对象
    const kernelInit =
      typeof this.kernelInitializer === 'string'
        ? tf.initializers[this.kernelInitializer]()
        : this.kernelInitializer;
    const biasInit = tf.initializers.zeros();

    // Q, K, V 的全连接投影权重
    this.qKernel = this.addWeight(
      'qKernel', [queryDim, keyProjectionDim], null, kernelInit);
    this.kKernel = this.addWeight(
      'kKernel', [keyInputDim, keyProjectionDim], null, kernelInit);
    this.vKernel = this.addWeight(
      'vKernel', [valueInputDim, valueProjectionDim], null, kernelInit);
    this.oKernel = this.addWeight(
      'oKernel', [valueProjectionDim, queryDim], null, kernelInit);

    if (this.useBias) {
      this.qBias = this.addWeight(
        'qBias', [keyProjectionDim], null, biasInit);
      this.kBias = this.addWeight(
        'kBias', [keyProjectionDim], null, biasInit);
      this.vBias = this.addWeight(
        'vBias', [valueProjectionDim], null, biasInit);
      this.oBias = this.addWeight('oBias', [queryDim], null, biasInit);
    }

    this.built = true;
  }

  computeOutputShape(inputShape) {
    // 输出形状与 query 相同: [batchSize, seqLenQ, dim]
    const [queryShape] = this.normalizeInputShapes(inputShape);
    return queryShape;
  }

  /**
   * @param {Tensor|Tensor[]} inputs - [query, key, value] 或仅 [query] 或 Tensor
   * @param {Object} kwargs
   * @param {Tensor} [kwargs.mask] - 可选 mask，形状 [batch, seqQ, seqK]
   */
  call(inputs, kwargs) {
    let [q, k, v] = this.normalizeInputs(inputs);
    const mask = kwargs && kwargs.mask;
    const outputDim = this.getFeatureDim(q.shape);
    const keyProjectionDim = this.getProjectionDim(this.keyDim);
    const valueProjectionDim = this.getProjectionDim(this.valueDim);

    // 线性投影
    q = this.projectInput(q, this.qKernel, this.qBias, keyProjectionDim);
    k = this.projectInput(k, this.kKernel, this.kBias, keyProjectionDim);
    v = this.projectInput(v, this.vKernel, this.vBias, valueProjectionDim);

    // 获取尺寸信息
    const seqLenQ = q.shape[SEQUENCE_AXIS];
    const headDimQ = this.keyDim;
    const headDimV = this.valueDim;

    // 重塑并转置以分头
    q = this.splitHeads(q, headDimQ);
    k = this.splitHeads(k, headDimQ);
    v = this.splitHeads(v, headDimV);

    // 缩放点积注意力
    let scores = tf.matMul(q, k, false, true);
    const scale = Math.sqrt(headDimQ);
    scores = tf.div(scores, tf.scalar(scale));
    if (mask) {
      const maskTensor = tf.cast(mask, 'float32');
      const addMask = tf.mul(tf.sub(tf.scalar(1.0), maskTensor), tf.scalar(MASK_FILL_VALUE));
      scores = tf.add(scores, addMask);
    }
    const attn = tf.softmax(scores, LAST_AXIS);

    // 加权求和
    let context = tf.matMul(attn, v);
    context = this.combineHeads(context, seqLenQ, headDimV);

    // 输出投影
    let output = this.projectInput(context, this.oKernel, this.oBias, outputDim);
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

// 注册类以支持序列化和模型保存/加载
tf.serialization.registerClass(MultiHeadAttention);




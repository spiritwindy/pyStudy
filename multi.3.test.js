import * as tf from '@tensorflow/tfjs';

// --------------------------------------------------
// 1. 多头注意力层 MultiHeadAttention 实现 (来自您之前的代码)
// --------------------------------------------------
export class MultiHeadAttention extends tf.layers.Layer {
  /**
   * @param {Object} config
   * @param {number} config.numHeads - 注意力头数
   * @param {number} config.keyDim - 每个头的 key/query 维度
   * @param {number} [config.valueDim] - 每个头的 value 维度 (默认与 keyDim 相同)
   * @param {boolean} [config.useBias] - 是否在投影层使用 bias
   * @param {string|tf.initializers.Initializer} [config.kernelInitializer] - 权重初始化方法或初始化器实例
   */
  constructor(config) {
    super(config);
    this.numHeads = config.numHeads;
    this.keyDim = config.keyDim;
    this.valueDim = config.valueDim || config.keyDim;
    this.useBias = config.useBias != null ? config.useBias : true;
    this.kernelInitializer = config.kernelInitializer || 'glorotUniform';
    this.supportsMasking = true; // 支持遮罩
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
    // 特征维度 (例如: 4)
    const dim = queryShape[queryShape.length - 1];

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
    // 输出投影权重 (将多头结果合并回原始维度 dim)
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
   * @param {Object} [kwargs]
   * @param {Tensor} [kwargs.mask] - 可选 mask，形状 [batch, seqQ, seqK] 或兼容的广播形状
   */
  call(inputs, kwargs) {
    let q, k, v;
    // 输入可以是单个张量 (q=k=v) 或包含 q, k, v 的数组
    if (Array.isArray(inputs)) {
      q = inputs[0];
      k = inputs[1] || q; // 如果未提供 k，则 k=q
      v = inputs[2] || k; // 如果未提供 v，则 v=k (或 v=q 如果 k 也未提供)
    } else {
      q = k = v = inputs;
    }
    // 提取可选的 mask
    const mask = kwargs && kwargs.mask;

    // --- 1. 线性投影 ---
    let query = tf.matMul(q, this.qKernel.read());
    let key = tf.matMul(k, this.kKernel.read());
    let value = tf.matMul(v, this.vKernel.read());
    if (this.useBias) {
      query = tf.add(query, this.qBias.read());
      key = tf.add(key, this.kBias.read());
      value = tf.add(value, this.vBias.read());
    }

    // --- 2. 分割头 & 重塑 ---
    const batchSize = query.shape[0];
    const seqLenQ = query.shape[1]; // 查询序列长度
    const seqLenK = key.shape[1];   // 键/值序列长度

    // reshape + transpose => [batchSize, numHeads, seqLen, headDim]
    query = this.splitHeads(query, batchSize, seqLenQ);
    key = this.splitHeads(key, batchSize, seqLenK);
    value = this.splitHeads(value, batchSize, seqLenK, this.valueDim); // 使用 valueDim

    // --- 3. 缩放点积注意力 ---
    let scores = tf.matMul(query, key, false, true); // (..., seqLenQ, headDimQ) * (..., seqLenK, headDimQ)^T => (..., seqLenQ, seqLenK)
    const scale = Math.sqrt(this.keyDim);
    scores = tf.div(scores, tf.scalar(scale));

    // 应用 mask (在 softmax 之前)
    if (mask) {
        // 确保 mask 维度兼容: 需要 [batch, 1, 1, seqLenK] 或 [batch, 1, seqLenQ, seqLenK]
        // 通常 mask 是 [batch, seqLenQ, seqLenK]，需要扩展维度以匹配 scores [batch, numHeads, seqLenQ, seqLenK]
        let attentionMask = tf.expandDims(mask, 1); // -> [batch, 1, seqLenQ, seqLenK]
        const addMask = tf.mul(tf.sub(tf.scalar(1.0), tf.cast(attentionMask, 'float32')), tf.scalar(-1e9));
        scores = tf.add(scores, addMask);
    }

    const attn = tf.softmax(scores, -1); // 在最后一个维度 (seqLenK) 上进行 softmax

    // --- 4. 加权求和 (输出上下文向量) ---
    let context = tf.matMul(attn, value); // (..., seqLenQ, seqLenK) * (..., seqLenK, headDimV) => (..., seqLenQ, headDimV)

    // --- 5. 合并头 & 输出投影 ---
    context = this.combineHeads(context, batchSize, seqLenQ); // -> [batchSize, seqLenQ, numHeads * valueDim]

    let output = tf.matMul(context, this.oKernel.read());
    if (this.useBias) {
      output = tf.add(output, this.oBias.read());
    }
    return output; // 形状: [batchSize, seqLenQ, dim]
  }

  // 辅助函数：分割头
  splitHeads(x, batchSize, seqLen, headDim = this.keyDim) {
      const numHeads = this.numHeads;
      // 从 [batchSize, seqLen, numHeads * headDim]
      // 变为 [batchSize, numHeads, seqLen, headDim]
      return tf.tidy(() => {
          const reshaped = tf.reshape(x, [batchSize, seqLen, numHeads, headDim]);
          return tf.transpose(reshaped, [0, 2, 1, 3]);
      });
  }

  // 辅助函数：合并头
  combineHeads(x, batchSize, seqLen) {
      const numHeads = this.numHeads;
      const headDim = this.valueDim; // 合并时使用 value 维度
      // 从 [batchSize, numHeads, seqLen, headDim]
      // 变为 [batchSize, seqLen, numHeads * headDim]
      return tf.tidy(() => {
          const transposed = tf.transpose(x, [0, 2, 1, 3]); // -> [batchSize, seqLen, numHeads, headDim]
          return tf.reshape(transposed, [batchSize, seqLen, numHeads * headDim]);
      });
  }

  getConfig() {
    const baseConfig = super.getConfig();
    return Object.assign({}, baseConfig, {
      numHeads: this.numHeads,
      keyDim: this.keyDim,
      valueDim: this.valueDim,
      useBias: this.useBias,
      kernelInitializer: this.kernelInitializer // 注意：如果是自定义初始化器实例，可能需要特殊处理
    });
  }

  static get className() {
    return 'MultiHeadAttention';
  }
}

// --------------------------------------------------
// 2. 构建堆叠注意力模型
// --------------------------------------------------
function buildStackedAttentionModel(seqLen, inputDim, outputDim, numAttentionLayers = 3, numHeads = 2, keyDim = 8, ffnDim = 32) {
  const inputs = tf.input({ shape: [seqLen, inputDim], name: 'input_sequence' }); // 输入形状: [批次大小, 24, 4]

  let x = inputs;

  // 堆叠 N 层 Transformer 块
  for (let i = 0; i < numAttentionLayers; i++) {
    // --- Transformer 块开始 ---
    const layerNameSuffix = `_${i + 1}`;

    // 块 1: 多头自注意力 +残差 & 层归一化
    const attention = new MultiHeadAttention({
        numHeads: numHeads,
        keyDim: keyDim,
        name: `multi_head_attention${layerNameSuffix}`
    });
    // 自注意力 (Q=K=V=x)
    const attentionOutput = attention.apply(x); // 输出形状仍然是 [批次大小, 24, 4]
    // 残差连接
    const x1 = tf.layers.add({name: `add_attention${layerNameSuffix}`}).apply([x, attentionOutput]);
    // 层归一化
    const norm1 = tf.layers.layerNormalization({name: `norm_attention${layerNameSuffix}`}).apply(x1);

    // 块 2: 前馈网络 + 残差 & 层归一化
    // 通常是一个扩展层 (ReLU 激活) + 一个投影回原始维度的层
    const ffn1 = tf.layers.dense({
        units: ffnDim, // 扩展维度
        activation: 'relu',
        name: `ffn_expand${layerNameSuffix}`
    }).apply(norm1);
    const ffn2 = tf.layers.dense({
        units: inputDim, // 投影回原始维度
        name: `ffn_project${layerNameSuffix}`
    }).apply(ffn1);
    // 残差连接
    const x2 = tf.layers.add({name: `add_ffn${layerNameSuffix}`}).apply([norm1, ffn2]);
    // 层归一化
    const norm2 = tf.layers.layerNormalization({name: `norm_ffn${layerNameSuffix}`}).apply(x2);
    // --- Transformer 块结束 ---

    // 更新 x 为当前块的输出，作为下一块的输入
    x = norm2;
  }

  // 聚合序列输出：使用全局平均池化将 [批次大小, 24, 4] 转换为 [批次大小, 4]
  const pooledOutput = tf.layers.globalAveragePooling1d({name: 'global_avg_pooling'}).apply(x);

  // (可选) 添加一个最终的 Dense 层进行最后的变换
  const finalOutput = tf.layers.dense({
      units: outputDim, // 输出维度为 4
      activation: 'linear', // 或者根据任务选择 'softmax', 'sigmoid' 等
      name: 'final_output_projection'
  }).apply(pooledOutput);

  // 创建并返回模型
  const model = tf.model({ inputs: inputs, outputs: finalOutput });
  return model;
}

// --------------------------------------------------
// 3. 使用示例
// --------------------------------------------------

// 定义模型参数
const SEQ_LENGTH = 24;
const INPUT_DIM = 4;
const OUTPUT_DIM = 4;
const NUM_ATTENTION_LAYERS = 3; // 堆叠 3 层
const NUM_HEADS = 2;            // 每个注意力层用 2 个头
const KEY_DIM = 8;              // 每个头的 Key/Query 维度
const FFN_HIDDEN_DIM = 32;      // 前馈网络的隐藏层维度

// 构建模型
const stackedModel = buildStackedAttentionModel(
    SEQ_LENGTH,
    INPUT_DIM,
    OUTPUT_DIM,
    NUM_ATTENTION_LAYERS,
    NUM_HEADS,
    KEY_DIM,
    FFN_HIDDEN_DIM
);

// 打印模型结构
stackedModel.summary();

// 创建一些虚拟数据进行测试 (批次大小为 2)
const batchSize = 2;
const dummyInput = tf.randomNormal([batchSize, SEQ_LENGTH, INPUT_DIM]);

// 进行预测
const prediction = stackedModel.predict(dummyInput);

console.log("\n输入形状:", dummyInput.shape);
console.log("预测输出形状:", prediction.shape);
console.log("预测输出示例 (第一条):");
prediction.slice([0, 0], [1, OUTPUT_DIM]).print(); // 打印第一个样本的预测结果

// 清理张量 (在实际应用中，模型训练和预测后可能需要)
// tf.dispose([dummyInput, prediction]);
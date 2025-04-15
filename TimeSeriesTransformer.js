// Transformer时间序列模型

const CONFIG = {
  SEQ_LENGTH: 24,    // 输入序列长度
  D_MODEL: 64,       // 模型维度
  N_HEADS: 4,        // 注意力头数
  N_LAYERS: 3,       // 编码器层数
  BATCH_SIZE: 32,
  EPOCHS: 40,
  LR: 1e-3,
  INPUT_DIM: 1,      // 输入特征维度
  OUTPUT_DIM: 1      // 输出特征维度
};
const tf = require('@tensorflow/tfjs');

const { MultiHeadAttention } = require('./multihead');

class TimeSeriesTransformer {
  constructor() {
    this.encoderLayers = Array(CONFIG.N_LAYERS).fill().map(() =>
      new MultiHeadAttention(CONFIG.N_HEADS, CONFIG.D_MODEL / CONFIG.N_HEADS)
    );

    this.positionEncoding = this.buildPositionEncoding();
    // 让 decoder 变成一个 trainable 层
    this.decoder = tf.sequential({
      layers: [
        tf.layers.dense({ units: 16, activation: 'relu', inputShape: [CONFIG.D_MODEL] }),
        tf.layers.dense({ units: CONFIG.OUTPUT_DIM })  // 输出维度改为 CONFIG.OUTPUT_DIM
      ]
    });
  }

  buildPositionEncoding() {
    const position = tf.range(CONFIG.SEQ_LENGTH, 0, -1)
      .reshape([1, CONFIG.SEQ_LENGTH, 1]);
    const divTerm = tf.exp(tf.range(0, CONFIG.D_MODEL, 2)
      .mul(-Math.log(10000.0) / CONFIG.D_MODEL));
    return position.mul(divTerm).sin();
  }

  encode(inputs) {
    const posEnc = this.positionEncoding.slice(
      [0, 0, 0],
      [1, inputs.shape[1], inputs.shape[2]]
    );
    return inputs.add(posEnc);
  }

  predict(inputs) {
    let x = this.encode(inputs);

    // 编码器层堆叠
    for (const layer of this.encoderLayers) {
      const attnOutput = layer.apply([x, x, x]);
      x = tf.layers.layerNormalization().apply(
        x.add(attnOutput)
      );
    }

    // 获取最后一个时间步
    const lastStep = x.slice(
      [0, x.shape[1] - 1, 0],
      [x.shape[0], 1, x.shape[2]]
    ).squeeze([1]);

    // 通过 `apply()` 确保 decoder 参与梯度计算
    return this.decoder.apply(lastStep);
  }
}

module.exports = { TimeSeriesTransformer, CONFIG };
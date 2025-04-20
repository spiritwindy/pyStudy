import * as tf from '@tensorflow/tfjs';
import {MultiHeadAttention} from "./multi.js"
// ====== 使用示例 ======
(async () => {
  const dim = 64;
  const input = tf.input({ shape: [10, dim] });
  const key = tf.input({ shape: [10, dim] });
  const val = tf.input({ shape: [10, dim] });

  const attn = new MultiHeadAttention({ numHeads: 8, keyDim: 16 });
  const output = attn.apply([input, key, val]);
  const model = tf.model({ inputs: [input, key, val], outputs: output });
  model.summary();

  // 测试前向
  const qData = tf.randomNormal([2, 10, dim]);
  const kData = tf.randomNormal([2, 10, dim]);
  const vData = tf.randomNormal([2, 10, dim]);
  const y = model.predict([qData, kData, vData]);
  // y.print(true);
  console.log(y.shape);
})();
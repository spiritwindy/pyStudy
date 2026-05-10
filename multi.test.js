import * as tf from '@tensorflow/tfjs';
import {MultiHeadAttention} from "./multi.js"
// ====== 使用示例 ======
(async () => {
  const numHeads = 8;
  const keyDim = 64;
  const dim = numHeads * keyDim;
  const input = tf.input({ shape: [10, dim] });
  const key = tf.input({ shape: [10, dim] });
  const val = tf.input({ shape: [10, dim] });

  const attn = new MultiHeadAttention({ numHeads, keyDim });
  const output = attn.apply([input, key, val]);
  const model = tf.model({ inputs: [input, key, val], outputs: output });
  model.summary();

  // 测试前向
  const qData = tf.randomNormal([2, 10, dim]);
  const kData = tf.randomNormal([2, 10, dim]);
  const vData = tf.randomNormal([2, 10, dim]);
  const y = model.predict([qData, kData, vData]);
  y.print();
  console.log(y.shape);
})();

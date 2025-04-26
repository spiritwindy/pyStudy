import * as tf from '@tensorflow/tfjs';
import {MultiHeadAttentionLayer } from "./multihead1.js"

function createAttentionModel() {
    const model = tf.sequential();
    const mhaLayer = new MultiHeadAttentionLayer(4, 64);
    // 构建模型结构（典型Transformer编码块）
    model.add(tf.layers.inputLayer({ inputShape: [32, 128] }));
    //     let p = tf.layers.inputLayer({ shape: [32, 128] });
    //   model.add(p); // 输入序列长度32，特征维度128
    model.add(mhaLayer );
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));

    return model;
}

// 2. 生成三维测试数据（batch_size=2, seq_len=32, features=128）
function generateTestData() {
  return {
    query: tf.randomNormal([2, 32, 128]),  // 随机生成query向量
    value: tf.randomNormal([2, 32, 128]),  // 随机生成value向量
    key: tf.randomNormal([2, 32, 128])     // 随机生成key向量
  };
}

// 3. 测试模型效果
async function testModel() {
  const model = createAttentionModel();
  const testData = generateTestData();
  
  // 编译模型（使用Adam优化器和分类交叉熵损失）
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  

}

// 执行测试
testModel().catch(console.error);
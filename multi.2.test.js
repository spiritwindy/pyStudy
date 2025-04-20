import * as tf from '@tensorflow/tfjs';
import {MultiHeadAttention} from "./multi.js"
function generateTestData(batchSize = 4, seqLen = 10, modelDim = 512, numClasses = 10) {
    const x = tf.randomNormal([batchSize, seqLen, modelDim]);
  
    const labelIndices = tf.randomUniform([batchSize, seqLen], 0, numClasses, 'int32');
    const y = tf.oneHot(labelIndices, numClasses); // shape: [batchSize, seqLen, numClasses]
    
    return { x, y };
  }
  
  
  async function test1(params) {
    // 堆叠两个Transformer块
    const model = tf.sequential({
      layers: [
        transformerBlock(512, 8),
        transformerBlock(512, 8),
        tf.layers.dense({ units: 10, activation: 'softmax' }) // 示例输出层
      ]
    });
  
    model.compile({
      optimizer: tf.train.adam(), // 使用 Adam 优化器
      loss: 'categoricalCrossentropy', // 损失函数为交叉熵
      metrics: ['accuracy'] // 评估指标为准确率
    });
    const { x, y } = generateTestData();
    
    console.log("x", x.shape)
    console.log("y", y.shape)
  
    model.summary();
    
    console.log('Training...');
    await model.fit(x, y, {
      epochs: 5,
      batchSize: 4,
      callbacks: tf.callbacks.earlyStopping({ patience: 2 })
    });
  
    console.log('Done training.');
  }
  test1();
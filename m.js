import * as tf from '@tensorflow/tfjs';

// 1. 创建模型
const model = tf.sequential();

// 2. 添加层
model.add(tf.layers.dense({
  inputShape: [3],   // 输入是3维
  units: 16,         // 隐藏层神经元数量（可以自己调整）
  activation: 'relu' // 激活函数
}));

model.add(tf.layers.dense({
  units: 1,          // 输出是1维
  activation: 'linear' // 回归问题用linear，分类问题可以用sigmoid或softmax
}));

// 3. 编译模型
model.compile({
  optimizer: tf.train.adam(0.01), // 优化器，学习率可以调整
  loss: 'meanSquaredError'        // 损失函数，回归任务用MSE
});

// 4. 生成一些随机数据进行训练示范
const xs = tf.randomNormal([100, 3]); // 100个样本，每个样本3维
const ys = tf.randomNormal([100, 1]); // 100个对应的输出

// 5. 训练模型
async function trainModel() {
  await model.fit(xs, ys, {
    epochs: 50,            // 训练轮数
    batchSize: 16,         // 批大小
    validationSplit: 0.2,  // 验证集比例
    callbacks: tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 5 })
  });

  console.log('训练完成！');

  // 测试一下模型
  const testInput = tf.tensor2d([[0.5, -0.2, 1.0]]);
  const prediction = model.predict(testInput);
  prediction.print();
}

trainModel();
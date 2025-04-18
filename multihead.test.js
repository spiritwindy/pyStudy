const tf = require('@tensorflow/tfjs');
const { MultiHeadAttention } = require('./multihead'); // 根据实际路径调整

test1();

function test1() {


    // 实例化 MultiHeadAttention 类
    const numHeads = 4;
    const keyDim = 64;
    const mha = new MultiHeadAttention(numHeads, keyDim);

    // 准备输入数据
    const batchSize = 1;
    const sequenceLength = 10;
    const featureDimension = 256; // 输入特征维度
    const query = tf.randomNormal([batchSize, sequenceLength, featureDimension]);
    const key = tf.randomNormal([batchSize, sequenceLength, featureDimension]);
    const value = tf.randomNormal([batchSize, sequenceLength, featureDimension]);

    // 使用多头注意力机制进行预测
    const output = mha.apply([query, key, value]);

    // 输出结果
    output.print();
}



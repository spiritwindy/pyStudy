import { MultiHeadAttentionLayer } from "./MultiHeadAttention1.js";
import { input, model as _model, randomNormal } from '@tensorflow/tfjs';
  // 定义测试函数
function test() {
    const sequenceLength = 10;
    const featureDim = 4;

    // 定义输入层
    const qInput = input({ shape: [sequenceLength, featureDim] });
    const kInput = input({ shape: [sequenceLength, featureDim] });
    const vInput = input({ shape: [sequenceLength, featureDim] });

    // 实例化多头注意力层（8个头，每个key维度32）
    const mhaLayer = new MultiHeadAttentionLayer(8, 32);
    const mhaOutput = mhaLayer.apply([qInput, kInput, vInput]);

    // 构建模型
    const model = _model({
        inputs: [qInput, kInput, vInput],
        outputs: mhaOutput
    });

    // 打印模型结构
    model.summary();

    // 生成随机训练数据
    function generateTrainingData(numSamples) {
        const qData = randomNormal([numSamples, sequenceLength, featureDim]);
        const kData = randomNormal([numSamples, sequenceLength, featureDim]);
        const vData = randomNormal([numSamples, sequenceLength, featureDim]);
        return { qData, kData, vData };
    }

    const { qData, kData, vData } = generateTrainingData(100);

    // 进行一次前向计算，确保模型正常工作
    model.predict([qData, kData, vData]).print();
}

// 调用测试函数
test();
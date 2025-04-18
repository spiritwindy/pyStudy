
const tf = require('@tensorflow/tfjs');
import _ from "tfjs-node-save"

import { fetchEarthquakes } from "./fetchData.js";

import "tfjs-node-save";
// import { MultiHeadAttentionLayer } from './multihead1';
import { MultiHeadAttention } from '@tensorflow/tfjs-layers/dist/layers/nlp/multihead_attention';
import { fetchEarthquakes } from './fetchData';

function createSequences(data, seqLength = 10) {
    const sequences = [];
    const labels = [];

    for (let i = 0; i < data.length - seqLength; i++) {
        const inputSeq = data.slice(i, i + seqLength);
        const label = [data[i + seqLength].time, data[i + seqLength].latitude, data[i + seqLength].longitude];
        sequences.push(inputSeq.map(d => [d.time, d.latitude, d.longitude, d.magnitude]));
        labels.push(label);
    }

    return { sequences, labels };
}



function createTransformerModel(inputSeqLength, featureDim, numHeads, keyDim) {
    // 定义输入层，输入形状为 [inputSeqLength, featureDim]
    const input = _input({ shape: [inputSeqLength, featureDim] });

    // 添加多头注意力层
    const mhaLayer = new MultiHeadAttention({ numHeads, keyDim });
    const attentionOutput = mhaLayer.call(input, { value: input });

    // 添加残差连接和归一化层
    const addAndNorm1 = layers.add().apply([input, attentionOutput]);
    const norm1 = layers.layerNormalization().apply(addAndNorm1);

    // 添加前馈网络
    const dense1 = layers.dense({ units: 128, activation: 'relu' }).apply(norm1);
    const dense2 = layers.dense({ units: 64, activation: 'relu' }).apply(dense1);

    // 添加残差连接和归一化层
    const addAndNorm2 = layers.add().apply([norm1, dense2]);
    const norm2 = layers.layerNormalization().apply(addAndNorm2);

    // 添加输出层，预测时间、纬度和经度
    const output = layers.dense({ units: 3 }).apply(norm2);

    // 构建模型
    const model = _model({ inputs: input, outputs: output });
    model.compile({
        optimizer: train.adam(), // 使用 Adam 优化器
        loss: 'meanSquaredError', // 损失函数为均方误差
        metrics: ['mae'] // 评估指标为平均绝对误差
    });

    return model; // 返回构建好的模型
}

async function trainTransformerModel() {
    // 获取地震数据
    let data = await fetchEarthquakes();
    const { sequences, labels } = createSequences(data);

    // 转换为张量
    const xs = tensor(sequences);
    const ys = tensor(labels);

    // 创建 Transformer 模型
    const inputSeqLength = sequences[0].length;
    const featureDim = sequences[0][0].length;
    const model = createTransformerModel(inputSeqLength, featureDim, 8, 32);

    // 训练模型
    await model.fit(xs, ys, {
        epochs: 50,
        batchSize: 32,
        validationSplit: 0.2
    });

    // 保存模型
    await model.save('file://./earthquake_transformer_model');
    console.log('模型已保存');
}

trainTransformerModel();
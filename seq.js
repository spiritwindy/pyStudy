// 导入 TensorFlow.js
import * as tf from '@tensorflow/tfjs';
import { tensor3d, tensor2d } from '@tensorflow/tfjs';
import { normalizeValues, denormalizeValues } from "./time.js";
import { fetchEarthquakes } from "./fetchData.js";
let CONFIG = {
    OUTPUT_DIM: 4,
    SEQ_LENGTH: 20
}

import "tfjs-node-save";

// 创建模型函数
function createModel() {
    const model = tf.sequential();

    // 添加 LSTM 层（可调整单元数）
    model.add(tf.layers.lstm({
        units: 64,
        inputShape: [CONFIG.SEQ_LENGTH, 4], // [时间步长, 特征维度]
        returnSequences: false // 只输出最后时间步的结果
    }));

    // 可选：添加更多层（根据需要调整）
    // model.add(tf.layers.dropout({ rate: 0.2 }));  // 防止过拟合
    // model.add(tf.layers.dense({ units: 32, activation: 'relu' }));

    // 输出层（假设回归任务）
    model.add(tf.layers.dense({
        units: 4,          // 输出维度（根据需求调整）
        activation: 'linear' // 回归任务使用线性激活
    }));

    // 编译模型
    model.compile({
        optimizer: tf.train.adam(0.001), // 学习率可调
        loss: 'meanSquaredError'         // 分类任务可改为 'categoricalCrossentropy'
    });

    return model;
}

// 初始化模型
const model = createModel();

// 查看模型结构
model.summary();



// 创建滑动窗口数据集
// 修改后的createDataset函数
/**
 * 
 * @param { {
    time: number;
    latitude: any;
    longitude: any;
    magnitude: any;
}[]} data 
 * @param {*} seqLength 
 * @returns 
 */

function createDataset(data, seqLength) {
    const X = [];
    const y = [];

    for (let i = 0; i < data.length - seqLength; i++) {
        // 每个时间步包装为一个数组，创建二维结构
        const seq = data.slice(i, i + seqLength).map(val => normalizeValues([val.time, val.latitude, val.longitude, val.magnitude]));//val.latitude,val.longitude,val.magnitude));//val.latitude,val.longitude,val.magnitude
        X.push(seq);
        let val = data[i + seqLength]
        y.push(normalizeValues([val.time, val.latitude, val.longitude, val.magnitude]));
    }

    return {
        X: tensor3d(X),  // 现在会自动推断形状
        y: tensor2d(y)
    };
}
async function main() {
    // 生成数据
    const rawData = await fetchEarthquakes(1000);
    console.log("原始数据:", rawData.length);
    const { X, y } = createDataset(rawData, CONFIG.SEQ_LENGTH);
    async function onEpochEnd(epoch, logs) {
        console.log(`Epoch ${epoch + 1}: loss = ${logs.loss}, accuracy = ${JSON.stringify(logs)}`);
        await model.save('file://my-model-e1');
    }
    // 训练模型
    model.fit(X, y, {
        epochs: 50, // 训练50个周期
        batchSize: 32, // 每次使用32个样本进行更新
        callbacks: {
            onEpochEnd: onEpochEnd,
        }
    }).then(async () => {
        await model.save('file://my-model-e1');
        console.log('模型训练完成');
    });

}
main();
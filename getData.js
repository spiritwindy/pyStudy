
const tf = require('@tensorflow/tfjs');
import _ from "tfjs-node-save"

import { fetchEarthquakes } from "./fetchData.js";


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



function createModel(seqLength, featureSize) {
    const input = tf.input({ shape: [seqLength, featureSize] });

    console.log({seqLength,featureSize})
    // 使用自定义 MultiHeadAttention
    const mha = new MultiHeadAttention(4, featureSize); // 4 个头，每个头的维度为 featureSize
    console.log("Input shape:", input.shape);
  const transformerOutput = mha.apply([input, input, input]);
   console.log("Transformer output shape:", transformerOutput.shape);

    const flatten = tf.layers.flatten().apply(transformerOutput);
    const dense1 = tf.layers.dense({ units: 128, activation: 'relu' }).apply(flatten);
    const output = tf.layers.dense({ units: 3 }).apply(dense1); // 预测 (时间、纬度、经度)
    const model = tf.model({ inputs: input, outputs: output });

    model.compile({
        optimizer: tf.train.adam(),
        loss: 'meanSquaredError'
    });

    return model;
}

async function test() {
    let data = await fetchEarthquakes();
    const { sequences, labels } = createSequences(data);
    const xs = tf.tensor(sequences).reshape([-1, sequences[0].length, sequences[0][0].length]);
    const ys = tf.tensor(labels);

    const model = createModel(sequences[0].length, sequences[0][0].length);
    await model.fit(xs, ys, {
        epochs: 50,
        batchSize: 32
    });

    // 保存模型
    await model.save('file://./earthquake_model');
}
test();

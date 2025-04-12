const tf = require('@tensorflow/tfjs');
const { TimeSeriesTransformer, CONFIG } = require('./TimeSeriesTransformer'); // 引入 TimeSeriesTransformer
const {fetchEarthquakes} = require('./fetchData'); // 引入 fetchData

require("tfjs-node-save");
function createSequences(data, seqLength = CONFIG.SEQ_LENGTH) {
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

async function trainAndPredict() {
    let data = await fetchEarthquakes();
    const { sequences, labels } = createSequences(data);

    const xs = tf.tensor(sequences).reshape([-1, CONFIG.SEQ_LENGTH, 4]); // 4 个特征 (time, latitude, longitude, magnitude)
    const ys = tf.tensor(labels);

    const transformer = new TimeSeriesTransformer();

    // 编译模型
    transformer.decoder.compile({
        optimizer: tf.train.adam(CONFIG.LR),
        loss: 'meanSquaredError'
    });

    // 训练模型
    await transformer.decoder.fit(xs, ys, {
        epochs: CONFIG.EPOCHS,
        batchSize: CONFIG.BATCH_SIZE
    });

    // 预测
    const testInput = xs.slice([0, 0, 0], [1, CONFIG.SEQ_LENGTH, 4]); // 使用第一条序列作为测试输入
    const encodedInput = transformer.encode(testInput); // 编码输入
    const lastStep = encodedInput.slice([0, encodedInput.shape[1] - 1, 0], [1, 1, encodedInput.shape[2]]).squeeze([1]); // 提取最后一个时间步
    const prediction = transformer.decoder.predict(lastStep); // 使用最后一个时间步进行预测

    prediction.print(); // 打印预测结果

    // 保存模型
    await transformer.decoder.save('file://./earthquake_transformer_model');
}

trainAndPredict();
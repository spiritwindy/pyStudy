// --- 首先，确保你已经引入了 TensorFlow.js ---
// 在 HTML 中:
// <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
//
// 或者在 Node.js 中:
// const tf = require('@tensorflow/tfjs-node'); // 或 tfjs-node-gpu
import * as tf from '@tensorflow/tfjs';
import {MultiHeadAttention} from "../multi.js"
// --- 模型参数定义 ---
const sequenceLength = 24; // 输入序列长度 T
const inputDim = 4;       // 输入特征维度 F_in
const outputDim = 4;      // 输出特征维度 F_out (预测下一个时间步)
const numHeads = 4;       // 注意力头的数量 (需要能被 dModel 整除)
const dModel = 128;       // 模型内部的隐藏维度 (Attention Key/Value/Query 维度) - 设置为 numHeads 的倍数
const dff = 256;          // 前馈网络 (FFN) 的内部维度
const numAttentionLayers = 1; // 堆叠的注意力层数
const dropoutRate = 0.1;  // Dropout 比率，用于正则化

// --- 构建模型 ---
function buildTimeSeriesAttentionModel() {
    const inputs = tf.input({ shape: [sequenceLength, inputDim], name: 'input_sequence' }); // 输入形状: [批次大小, 序列长度, 特征维度]

    // --- 1. 输入投影 (可选但推荐) ---
    // 将输入特征维度映射到模型的内部维度 dModel
    let x = tf.layers.dense({
        units: dModel,
        name: 'input_projection'
    }).apply(inputs); // 输出: [批次大小, 24, dModel]

    // --- 注意: 实际应用中强烈推荐加入位置编码 (Positional Encoding) ---
    // 由于注意力机制本身不包含位置信息，需要手动添加。
    // 这里为了简化代码省略了位置编码，但在实际任务中必须添加。
    // 你可以使用固定的正弦/余弦编码或可学习的位置嵌入。
    // x = addPositionalEncoding(x, sequenceLength, dModel); // 假设有这个函数

    // --- 2. 堆叠的注意力层 ---
    for (let i = 0; i < numAttentionLayers; i++) {
        // --- 2a. 多头自注意力 (Multi-Head Self-Attention) ---
        const attentionOutput = new MultiHeadAttention({
            numHeads: numHeads, keyDim: dModel / numHeads ,
            name: `multi_head_attention_${i + 1}`
            // 注意: TensorFlow.js 的 multiHeadAttention 默认 Q, K, V 来自同一个输入
        }).apply([x,x,x]); // 输出: [批次大小, 24, dModel]

        // --- 2b. Add & Norm (残差连接 + 层归一化) ---
        let sublayer1Output = tf.layers.add({ name: `add_norm_1_add_${i + 1}` }).apply([x, attentionOutput]);
        sublayer1Output = tf.layers.layerNormalization({ name: `add_norm_1_norm_${i + 1}` }).apply(sublayer1Output);
        // 可选: 在残差连接后应用 Dropout
        sublayer1Output = tf.layers.dropout({ rate: dropoutRate, name: `add_norm_1_dropout_${i + 1}` }).apply(sublayer1Output);


        // --- 2c. 前馈网络 (Feed Forward Network) ---
        let ffnOutput = tf.layers.dense({
            units: dff,
            activation: 'relu', // 通常使用 ReLU 或 GeLU
            name: `ffn_dense_1_${i + 1}`
        }).apply(sublayer1Output);
        ffnOutput = tf.layers.dense({
            units: dModel, // 映射回 dModel
            name: `ffn_dense_2_${i + 1}`
        }).apply(ffnOutput);

        // --- 2d. Add & Norm (残差连接 + 层归一化) ---
        let sublayer2Output = tf.layers.add({ name: `add_norm_2_add_${i + 1}` }).apply([sublayer1Output, ffnOutput]);
        sublayer2Output = tf.layers.layerNormalization({ name: `add_norm_2_norm_${i + 1}` }).apply(sublayer2Output);
        // 可选: 在残差连接后应用 Dropout
        sublayer2Output = tf.layers.dropout({ rate: dropoutRate, name: `add_norm_2_dropout_${i + 1}` }).apply(sublayer2Output);

        // 更新 x 以便输入到下一层
        x = sublayer2Output; // 输出: [批次大小, 24, dModel]
    }

    // --- 3. 输出处理 ---
    // 目标是输出一个 [批次大小, outputDim] 的张量，代表下一个时间步的预测
    // 方法1: 使用全局平均池化聚合序列信息
    let output = tf.layers.globalAveragePooling1d({ name: 'global_avg_pooling' }).apply(x); // 输入 [批次, 24, dModel], 输出 [批次, dModel]

    // 方法2: 只取序列最后一个时间步的输出 (如果模型倾向于将最终预测集中在最后一步)
    // let output = tf.layers.cropping1D({cropping: [sequenceLength - 1, 0], name:'select_last_step'}).apply(x); // 输出 [批次, 1, dModel]
    // output = tf.layers.flatten().apply(output); // 展平为 [批次, dModel]

    // --- 4. 最终输出层 ---
    // 添加一个 Dense 层将 dModel 映射到最终的输出维度 outputDim
    output = tf.layers.dense({
        units: outputDim,
        activation: 'linear', // 对于回归预测，通常使用线性激活
        name: 'output_layer'
    }).apply(output); // 输出: [批次大小, outputDim]

    // --- 创建并返回模型 ---
    const model = tf.model({ inputs: inputs, outputs: output });
    return model;
}

// --- 模型使用示例 ---

// 1. 构建模型
const model = buildTimeSeriesAttentionModel();
model.summary(); // 打印模型结构

// 2. 编译模型
//    选择优化器 (adam 很常用) 和损失函数 (对于回归任务，常用均方误差)
model.compile({
    optimizer: tf.train.adam(),
    loss: 'meanSquaredError' // 或者 'meanAbsoluteError' 等
});

console.log("模型构建和编译完成.");

// --- 准备数据 (示例) ---
// ！！！你需要用你自己的真实数据替换这里的伪数据！！！
async function trainModel() {
    console.log("准备训练数据...");
    // 生成一些随机的训练数据作为示例
    const batchSize = 32;
    const numSamples = 100; // 训练样本数量 (实际中需要更多)

    // 生成伪输入数据 X: [numSamples, sequenceLength, inputDim]
    const trainX = tf.randomNormal([numSamples, sequenceLength, inputDim]);

    // 生成伪输出数据 Y: [numSamples, outputDim] (假设是序列之后的一个时间步)
    const trainY = tf.randomNormal([numSamples, outputDim]);

    console.log("开始训练...");
    const history = await model.fit(trainX, trainY, {
        batchSize: batchSize,
        epochs: 10, // 训练轮数 (需要根据实际情况调整)
        validationSplit: 0.2, // 使用一部分数据作为验证集
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, val_loss = ${logs.val_loss.toFixed(4)}`);
            }
        }
    });
    console.log("训练完成.");
    trainX.dispose();
    trainY.dispose();
    return history;
}

// --- 进行预测 (示例) ---
async function makePrediction() {
    console.log("准备预测数据...");
    // 生成一个或多个用于预测的输入序列
    const numPredictions = 5;
    const predictX = tf.randomNormal([numPredictions, sequenceLength, inputDim]); // [批次大小, 24, 4]

    console.log("进行预测...");
    const predictions = model.predict(predictX); // 输出形状: [批次大小, outputDim]

    console.log("预测结果 (形状):", predictions.shape);
    console.log("预测结果 (前几个):");
    predictions.print();

    predictX.dispose();
    predictions.dispose();
}

// --- 运行示例 ---
(async () => {
    await trainModel();
    await makePrediction();
    console.log("所有操作完成.");
})();

// --- 重要提示和改进方向 ---
// 1.  **位置编码 (Positional Encoding):** 这是基于 Transformer/Attention 模型处理序列数据的关键部分，代码中已注释其必要性。你需要实现并添加它。可以通过正弦/余弦函数生成固定编码，或使用 `tf.layers.embedding` 创建可学习的位置嵌入。
// 2.  **数据预处理:** 实际应用中，时间序列数据通常需要归一化（Normalization）或标准化（Standardization），这对模型训练至关重要。
// 3.  **超参数调优:** `dModel`, `numHeads`, `dff`, `numAttentionLayers`, `dropoutRate`, 学习率等都需要根据你的具体数据和任务进行调整。
// 4.  **损失函数和评估指标:** 根据你的具体预测目标（例如，预测值的范围、对误差的容忍度）选择合适的损失函数和评估指标。
// 5.  **模型变体:** 可以探索不同的输出处理方式（如取最后时间步输出而非全局池化）、不同的注意力机制变体（如带掩码的注意力，如果用于生成任务）或结合卷积层（CNN）等。
// 6.  **错误处理和资源管理:** 在实际应用中添加更健壮的错误处理，并确保使用 `tf.tidy` 或手动调用 `.dispose()` 来管理 Tensor 内存，防止泄漏。
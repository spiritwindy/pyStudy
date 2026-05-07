import { tensor3d, tensor2d, train, dispose, loadLayersModel } from '@tensorflow/tfjs';
import { mkdir, rm } from 'fs/promises';
import { TimeSeriesTransformer, CONFIG } from './TimeSeriesTransformer.js';
import { fetchEarthquakes } from "./earthquake.js";
import { normalizeValues, normalizeNextEventTarget, decodeNextEventPrediction } from "./time.js";
CONFIG.OUTPUT_DIM = 4;
CONFIG.EPOCHS = 20;
import "tfjs-node-save";
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
      const seq = data.slice(i, i + seqLength).map(val => normalizeValues([val.time,val.latitude,val.longitude,val.magnitude]));//val.latitude,val.longitude,val.magnitude));//val.latitude,val.longitude,val.magnitude
      X.push(seq); // [seqLength,4]
      let last = data[i + seqLength - 1];
      let val = data[i + seqLength];
      y.push(normalizeNextEventTarget(last, val));
    }
    
    return {
      X: tensor3d(X),  // 现在会自动推断形状
      y: tensor2d(y, [y.length, CONFIG.OUTPUT_DIM])
    };
  }



// 训练流程
async function main() {
  // 生成数据
  const rawData = await fetchEarthquakes(1000);
  console.log("原始数据:", rawData.length);
  const { X, y } = createDataset(rawData, CONFIG.SEQ_LENGTH);


  // 划分训练集/验证集
  const splitIdx = Math.floor(X.shape[0] * 0.8);
  console.log("X",X.shape)
  console.log("Y",y.shape)
  const X_train = X.slice([0, 0, 0], [splitIdx, CONFIG.SEQ_LENGTH, CONFIG.OUTPUT_DIM]);
  const X_val = X.slice([splitIdx, 0, 0], [-1, CONFIG.SEQ_LENGTH, CONFIG.OUTPUT_DIM]);
  console.log("X_train",X_train.shape)
  console.log("X_val",X_val.shape)

  const y_train = y.slice([0, 0], [splitIdx, CONFIG.OUTPUT_DIM]);
  const y_val = y.slice([splitIdx, 0], [-1, CONFIG.OUTPUT_DIM]);
  const X_val_last = X_val
    .slice([0, CONFIG.SEQ_LENGTH - 1, 0], [-1, 1, CONFIG.OUTPUT_DIM])
    .squeeze([1]);
  console.log("y_train",y_train.shape)
  console.log("y_val",y_val.shape)
  console.log("X_val_last", X_val_last.shape)
  // 初始化模型
  const model = new TimeSeriesTransformer();
  // await loadModelWeights(model); // 加载模型权重
  const optimizer = train.adam(CONFIG.LR);

  const lossFn = (yTrue, yPred) => yTrue.sub(yPred).square().mean();

  /**
   * 
   * @param {import('@tensorflow/tfjs').Tensor2D} lastSteps
   * @param {import('@tensorflow/tfjs').Tensor2D} yTrue 
   * @param {import('@tensorflow/tfjs').Tensor2D} yPred 
   * @returns 
   */
  async function showDiff(lastSteps, yTrue, yPred) {
    const lastStepArray = await lastSteps.array();
    const predArray = await yPred.array(); // 转为普通数组
    const yTrueArray = await yTrue.array(); // 转为普通数组

    const valPredArray = predArray;
    for (var t = 0; t < Math.min(valPredArray.length, 5); t++) {
      let ele = valPredArray[t];
      let eleTrue = yTrueArray[t];
      let lastStep = lastStepArray[t];
      let resTure = decodeNextEventPrediction(eleTrue, lastStep);
      let res = decodeNextEventPrediction(ele, lastStep);
      console.log("-----------")
      console.log("pre",new Date(res[0]).toLocaleString(), res[1], res[2], res[3])
      console.log("actual", new Date(resTure[0]).toLocaleString(), resTure[1], resTure[2], resTure[3])
      
    }

    return yTrue.sub(yPred).square().mean();
  }

  console.log('开始训练...');
  for (let epoch = 0; epoch < CONFIG.EPOCHS; epoch++) {
    let totalLoss = 0;
    
    // 分批训练
    for (let i = 0; i <= X_train.shape[0] - CONFIG.BATCH_SIZE; i += CONFIG.BATCH_SIZE) {
      const batchX = X_train.slice([i, 0, 0], [CONFIG.BATCH_SIZE, CONFIG.SEQ_LENGTH, CONFIG.OUTPUT_DIM]);
      const batchY = y_train.slice([i, 0], [CONFIG.BATCH_SIZE, CONFIG.OUTPUT_DIM]);
      const batchLastStep = batchX
        .slice([0, CONFIG.SEQ_LENGTH - 1, 0], [CONFIG.BATCH_SIZE, 1, CONFIG.OUTPUT_DIM])
        .squeeze([1]);
      // console.log("batchX",batchX.shape)
      const loss = optimizer.minimize(() => {
       
        const pred = model.decoder.apply(batchLastStep);
        
        return lossFn(batchY, pred);
      }, true);
    
      totalLoss += loss.dataSync()[0];
      dispose([batchX, batchY, batchLastStep, loss]);
    }
    console.log("优化完成>>>>>>>>>>>>>>>>>>>>>>>>")
    // 验证损失
    const valPred = model.decoder.predict(X_val_last);
    valPred.print();
    console.log(valPred.shape)
    const valLoss = lossFn(y_val, valPred).dataSync()[0];
    await showDiff(X_val_last, y_val, valPred)
    console.log(
      `Epoch ${epoch + 1}/${CONFIG.EPOCHS} | ` +
      `Train Loss: ${(totalLoss/(X_train.shape[0]/CONFIG.BATCH_SIZE)).toFixed(4)} | ` +
      `Val Loss: ${valLoss.toFixed(4)}`
    );
    dispose(valPred);
    await saveModel(model);
  }
  console.log("开始连续预测未来的 30 次地震...");
  const futurePredictions = await predictFuture(model, X_val, 30);
  console.log("未来 30 次地震预测结果:", futurePredictions);


  // 保存模型
 
  console.log('模型已保存到 model/');
}

// 连续预测未来的 30 次地震
async function predictFuture(model, X_val, steps = 30) {
  let predictions = [];
  let currentInput = X_val.slice([0, 0, 0], [1, CONFIG.SEQ_LENGTH, CONFIG.OUTPUT_DIM]); // 从验证集取一个初始输入

  for (let i = 0; i < steps; i++) {
    const lastStep = currentInput
      .slice([0, CONFIG.SEQ_LENGTH - 1, 0], [1, 1, CONFIG.OUTPUT_DIM])
      .squeeze([1]);
    const pred = model.decoder.predict(lastStep); // 预测下一步
    const predArray = await pred.array(); // 转为普通数组
    const lastStepArray = await lastStep.array();
    const res = decodeNextEventPrediction(predArray[0], lastStepArray[0]);
    console.log(new Date(res[0]).toLocaleString(), res[1], res[2], res[3])

    predictions.push(res); // 保存预测结果

    // 更新输入，将预测结果作为下一次的输入
    const oldInput = currentInput;
    const nextInput = currentInput.slice([0, 1, 0], [1, CONFIG.SEQ_LENGTH - 1, CONFIG.OUTPUT_DIM]);
    const nextEvent = tensor3d([[normalizeValues([res[0], res[1], res[2], res[3]])]], [1, 1, CONFIG.OUTPUT_DIM]);
    currentInput = nextInput.concat(nextEvent, 1); // 拼接新的预测值

    dispose([oldInput, pred, lastStep, nextInput, nextEvent]); // 释放内存
  }

  dispose(currentInput);
  return predictions;
}


// 模型保存
/**
 * 
 * @param {TimeSeriesTransformer} model 
 */
async function saveModel(model) {

  await mkdir('model', { recursive: true });
  await rm('model/decoder', { recursive: true, force: true });

  let w = await model.decoder.save("file://./model/decoder")

}

/**
 * 加载模型权重
 * @param {TimeSeriesTransformer} model
 */
async function loadModelWeights(model) {
  model.decoder = await loadLayersModel("file://./model/decoder/model.json");

}


// 执行训练
main();

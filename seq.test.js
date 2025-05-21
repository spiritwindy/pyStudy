// 导入 TensorFlow.js
import * as tf from '@tensorflow/tfjs';
import { tensor3d, tensor2d, dispose, } from '@tensorflow/tfjs';
import { normalizeValues, denormalizeValues } from "./time.js";
import { fetchEarthquakes } from "./fetchData.js";
let CONFIG = {
  OUTPUT_DIM: 4,
  SEQ_LENGTH: 20
}

import "tfjs-node-save";


// 连续预测未来的 30 次地震
async function predictFuture(model, X_val, steps = 30) {

  let predictions = [];
  let currentInput = X_val.slice([0, 0, 0], [1, CONFIG.SEQ_LENGTH, CONFIG.OUTPUT_DIM]); // 从验证集取一个初始输入

  for (let i = 0; i < steps; i++) {
    const pred = model.predict(currentInput); // 预测下一步
    const predArray = await pred.array(); // 转为普通数组

    const valPredArray = predArray;
    for (var t = 0; t < valPredArray.length; t++) {
      let ele = valPredArray[t];
      let res = denormalizeValues(ele);
      console.log(new Date(res[0]).toLocaleString(), res[1], res[2], res[3])
    }

    predictions.push(predArray[0]); // 保存预测结果

    // 更新输入，将预测结果作为下一次的输入
    const nextInput = currentInput.slice([0, 1, 0], [1, CONFIG.SEQ_LENGTH - 1, CONFIG.OUTPUT_DIM]);
    currentInput = nextInput.concat(pred.reshape([1, 1, CONFIG.OUTPUT_DIM]), 1); // 拼接新的预测值

    dispose([pred, nextInput]); // 释放内存
  }

  return predictions;
}

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
  const model = await tf.loadLayersModel('file://./my-model-e1/model.json');
  // 生成数据
  const rawData = await fetchEarthquakes(1000);
  console.log("原始数据:", rawData.length);
  const { X, y } = createDataset(rawData, CONFIG.SEQ_LENGTH);

  const X_val = X.slice([X.shape[0] - 1, 0, 0], [-1, CONFIG.SEQ_LENGTH, CONFIG.OUTPUT_DIM]);
  console.log(X_val.shape);
  var x = await X_val.array()
  predictFuture(model, X_val)
}
main()
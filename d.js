import { tensor3d, tensor2d, train, dispose } from '@tensorflow/tfjs';
import { mkdir, writeFile } from 'fs/promises';
import { TimeSeriesTransformer, CONFIG } from './TimeSeriesTransformer.js';
import { fetchEarthquakes } from "./fetchData.js";
import { normalizeTimestamp,denormalizeTimestamp } from "./time.js";
CONFIG.OUTPUT_DIM =4;
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
      const seq = data.slice(i, i + seqLength).map(val => [ normalizeTimestamp (val.time),val.latitude,val.longitude,val.magnitude]);//val.latitude,val.longitude,val.magnitude
      X.push(seq);  // 现在X是number[][][]
      let val = data[i + seqLength]
      y.push( [normalizeTimestamp(val.time),val.latitude,val.longitude,val.magnitude]);
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
  console.log("y_train",y_train.shape)
  console.log("y_val",y_val.shape)
  // 初始化模型
  const model = new TimeSeriesTransformer();
  const optimizer = train.adam(CONFIG.LR);
  const lossFn = (yTrue, yPred) => yTrue.sub(yPred).square().mean();

  console.log('开始训练...');
  for (let epoch = 0; epoch < CONFIG.EPOCHS; epoch++) {
    let totalLoss = 0;
    
    // 分批训练
    for (let i = 0; i <= X_train.shape[0] - CONFIG.BATCH_SIZE; i += CONFIG.BATCH_SIZE) {
      const batchX = X_train.slice([i, 0, 0], [CONFIG.BATCH_SIZE, CONFIG.SEQ_LENGTH, CONFIG.OUTPUT_DIM]);
      const batchY = y_train.slice([i, 0], [CONFIG.BATCH_SIZE, CONFIG.OUTPUT_DIM]);
      console.log("batchX",batchX.shape)
      const loss = optimizer.minimize(() => {
        const pred = model.predict(batchX);
        return lossFn(batchY, pred);
      }, true);
    
      totalLoss += loss.dataSync()[0];
      dispose([batchX, batchY, loss]);
    }
    console.log("优化完成>>>>>>>>>>>>>>>>>>>>>>>>")
    // 验证损失
    const valPred = model.predict(X_val);
    valPred.print();
    console.log(valPred.shape)
    const valLoss = lossFn(y_val, valPred).dataSync()[0];
    
    console.log(
      `Epoch ${epoch + 1}/${CONFIG.EPOCHS} | ` +
      `Train Loss: ${(totalLoss/(X_train.shape[0]/CONFIG.BATCH_SIZE)).toFixed(4)} | ` +
      `Val Loss: ${valLoss.toFixed(4)}`
    );
    
    dispose(valPred);
  }

  // 保存模型
  await saveModel(model);
  console.log('模型已保存到 model/');
}

// 模型保存
async function saveModel(model) {
  const modelInfo = {
    positionEncoding: await model.positionEncoding.array(),
    decoderWeights: await model.decoder.getWeights()
  };
  await mkdir('model', { recursive: true });
  await writeFile('model/metadata.json', JSON.stringify({
    seqLength: CONFIG.SEQ_LENGTH,
    dModel: CONFIG.D_MODEL
  }));
  await writeFile('model/weights.bin', 
    new Uint8Array(await model.decoder.getWeights()[0].data()));
}

// 执行训练
main();
import * as tf from '@tensorflow/tfjs';
import { MultiHeadAttention } from '@tensorflow/tfjs-layers/dist/layers/nlp/multihead_attention';
// const  = pkg;
// 创建一个简单的模型，包含 MultiHeadAttention 层
const createModel = () => {
  const inputQuery = tf.input({ shape: [8, 16], name: 'query' }); // 输入查询张量
  const inputValue = tf.input({ shape: [8, 16], name: 'value' }); // 输入值张量

  // 实例化 MultiHeadAttention 层
  const attentionLayer = new MultiHeadAttention({
    numHeads: 4,          // 注意力头的数量
    keyDim: 16,           // 每个头的键的维度
    valueDim: 16,         // 每个头的值的维度
    dropout: 0.1,         // Dropout 概率
    useBias: true,        // 是否使用偏置
  });

  // 应用 MultiHeadAttention 层
  const [attentionOutput] = attentionLayer.apply([inputQuery, inputValue],{value: inputValue});

  // 创建模型
  const model = tf.model({
    inputs: [inputQuery, inputValue],
    outputs: attentionOutput,
    name: 'multi_head_attention_model',
  });

  return model;
};

// 创建并打印模型
const model = createModel();
model.summary();
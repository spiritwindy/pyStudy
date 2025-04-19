import * as tf from '@tensorflow/tfjs';
// 定义一个自定义层（继承自 tf.layers.Layer）
class CustomMultiplyLayer extends tf.layers.Layer {
    constructor(config) {
      super(config);
      this.factor = config.factor || 1.0;
    }
  
    computeOutputShape(inputShape) {
      return inputShape;
    }
  
    call(input, kwargs) {
      const x = Array.isArray(input) ? input[0] : input;
      return x.mul(tf.scalar(this.factor));
    }
  
    getConfig() {
      const config = super.getConfig();
      Object.assign(config, { factor: this.factor });
      return config;
    }
  
    static get className() {
      return 'CustomMultiplyLayer';
    }
}
const layer = new CustomMultiplyLayer({ factor: 3.0 });
const input = tf.tensor2d([[1, 2], [3, 4]]);
const output = layer.apply(input);
output.print();  // [[3,6],[9,12]]
tf.serialization.registerClass(CustomMultiplyLayer);
  
const model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [2], units: 2 }));
model.add(new CustomMultiplyLayer({ factor: 10 }));

model.summary();

const inputTensor = tf.tensor2d([[0.5, -0.2]]);
model.predict(inputTensor).print();
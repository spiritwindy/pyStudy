import { createRequire } from 'module'
import { resolve } from "import-meta-resolve";
import { fileURLToPath, pathToFileURL } from 'url';
import { Console } from 'console';
// const require = createRequire(process.cwd())
// const { exec } = require('child_process');

// let t= require.resolve('@tensorflow/tfjs-layers/dist/layers/nlp/multihead_attention.js')

// let p = import("@tensorflow/tfjs-layers/dist/layers/nlp/multihead_attention.js")
// console.log(p)

console.log(import.meta.resolve("@tensorflow/tfjs-layers/dist/layers/nlp/multihead_attentio","E:\\pyStudy"))

// console.log(import.meta.url)
console.log( resolve("@tensorflow/tfjs-layers/dist/layers/nlp/multihead_attentio", pathToFileURL("E:\\pyStudy\\")))
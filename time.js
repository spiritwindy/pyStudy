let min = new Date("1900-01-01").getTime() / 1000;
let max = new Date("2100-01-01").getTime() / 1000;

/**
 * 归一化时间戳
 * @param {number} timestamp - 要归一化的时间戳（毫秒）
 * @param {number} minTimestamp - 范围内的最小时间戳
 * @param {number} maxTimestamp - 范围内的最大时间戳
 * @returns {number} 归一化后的值（在0到1之间）
 */
export function normalizeTimestamp(timestamp, minTimestamp = min, maxTimestamp = max) {
    if (maxTimestamp === minTimestamp) {
        throw new Error("最大时间戳和最小时间戳不能相同");
    }
    return (timestamp - minTimestamp) / (maxTimestamp - minTimestamp);
}

/**
 * 反归一化，将归一化值转换回原始时间戳
 * @param {number} normalized - 归一化的值（0到1之间）
 * @param {number} minTimestamp - 范围内的最小时间戳
 * @param {number} maxTimestamp - 范围内的最大时间戳
 * @returns {number} 原始时间戳（毫秒）
 */
export function denormalizeTimestamp(normalized, minTimestamp = min, maxTimestamp = max) {
    // if (normalized < 0 || normalized > 1) {
    //     throw new Error("归一化值必须在0到1之间");
    // }
    return normalized * (maxTimestamp - minTimestamp) + minTimestamp;
}

// test();
 function test() {
    let t = Date.now();
    let res = normalizeTimestamp(t / 1000);
    console.log(res);
    console.log(new Date(denormalizeTimestamp(res) * 1000).toLocaleString());
}


// export default {normalizeTimestamp,denormalizeTimestamp}
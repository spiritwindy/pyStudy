let min = new Date("1900-01-01").getTime();
let max = new Date("2100-01-01").getTime();

// 归一化函数
export function normalizeValue(value, minValue, maxValue) {
    if (maxValue === minValue) {
        throw new Error("最大值和最小值不能相同");
    }
    return (value - minValue) / (maxValue - minValue);
}

// 反归一化函数
export function denormalizeValue(normalized, minValue, maxValue) {
    return normalized * (maxValue - minValue) + minValue;
}

const normalizeCase = [
    { minValue: min, maxValue: max, description: "Time" },
    { minValue: -90, maxValue: 90, description: "Latitude" },
    { minValue: -180, maxValue: 180, description: "Longitude" },
    { minValue: 0, maxValue: 11, description: "Magnitude" }
];
// 测试用例
export function normalizeValues(arr) {
    let res = new Array(arr.length)
    for (let index = 0; index < normalizeCase.length; index++) {
        res[index] = normalizeValue(arr[index], normalizeCase[index].minValue, normalizeCase[index].maxValue)
    }
    return res
}
// 测试用例
export function denormalizeValues(arr) {
    let res = new Array(arr.length)
    for (let index = 0; index < normalizeCase.length; index++) {
        res[index] = denormalizeValue(arr[index], normalizeCase[index].minValue, normalizeCase[index].maxValue)
    }
    return res
}
export function test() {
    // 调用测试函数
    const arr = [new Date().getTime(), Math.random() * 90, Math.random() * 180, 5]
    console.log(arr)
    let res = normalizeValues(arr);
    console.log(res);
    let origin = denormalizeValues(res);
    console.log(origin)

    // export default {normalizeTimestamp,denormalizeTimestamp}
}

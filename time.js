let min = new Date("1900-01-01").getTime();
let max = new Date("2100-01-01").getTime();
export const TIME_GAP_MAX_MS = 365 * 24 * 60 * 60 * 1000;

function clamp01(value) {
    if (!Number.isFinite(value)) {
        return 0;
    }
    return Math.min(1, Math.max(0, value));
}

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

export function normalizeTimeGap(gapMs) {
    return clamp01(gapMs / TIME_GAP_MAX_MS);
}

export function denormalizeTimeGap(normalizedGap) {
    return clamp01(normalizedGap) * TIME_GAP_MAX_MS;
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

export function normalizeNextEventTarget(lastEvent, nextEvent) {
    const normalized = normalizeValues([nextEvent.time, nextEvent.latitude, nextEvent.longitude, nextEvent.magnitude]);
    normalized[0] = normalizeTimeGap(nextEvent.time - lastEvent.time);
    return normalized;
}

export function decodeNextEventPrediction(prediction, lastNormalizedEvent) {
    const [lastTime] = denormalizeValues(lastNormalizedEvent);
    const [, latitude, longitude, magnitude] = denormalizeValues([
        0,
        clamp01(prediction[1]),
        clamp01(prediction[2]),
        clamp01(prediction[3])
    ]);
    const timeGap = Math.max(1, denormalizeTimeGap(prediction[0]));
    return [lastTime + timeGap, latitude, longitude, magnitude];
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

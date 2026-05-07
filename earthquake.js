import { queryEarthquakes, queryRange } from "./sqlite.js";

export async function fetchEarthquakes() {
    try {
        const rows = queryEarthquakes(6);
        const data = rows.map((row) => ({
            time: new Date(row.time).getTime(),
            latitude: row.latitude,
            longitude: row.longitude,
            magnitude: row.magnitude
        }));

        console.log(rows.length, "条地震数据");
        return data;
    } catch (error) {
        console.error("查询失败:", error.message);
    }
}

export async function getRang() {
    const range = queryRange();
    console.log(range);
    return range;
}
fetchEarthquakes()
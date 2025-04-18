import {  Op,fn,col } from "sequelize";
import { Earthquake } from "./sqlite.js"; // 引入定义的模型



export async function fetchEarthquakes() {
    try {
        // 查询所有地震数据并按时间排序
        const rows = await Earthquake.findAll({
            attributes: ["time", "latitude", "longitude", "magnitude"],
            order: [["time", "ASC"]],
            where: {
                magnitude: {
                    [Op.gte]: 7.5 // 最小震级为8.2
                }
            }
        });
        const data = rows.map(row => ({
            time: new Date(row.time).getTime(), // 转成秒
            latitude: row.latitude,
            longitude: row.longitude,
            magnitude: row.magnitude
        }));
        console.log(rows.length, "条地震数据:"); // 打印查询结果
        return data;
    } catch (error) {
        console.error("查询失败:", error.message);
    }
}

export async function getRang(params) {
            // 查询范围
    const range = await Earthquake.findAll({
        attributes: [
            [fn("MIN", col("time")), "minTime"],
            [fn("MAX", col("time")), "maxTime"],
            [fn("MIN", col("latitude")), "minLatitude"],
            [fn("MAX", col("latitude")), "maxLatitude"],
            [fn("MIN", col("longitude")), "minLongitude"],
            [fn("MAX", col("longitude")), "maxLongitude"],
            [fn("MIN", col("magnitude")), "minMagnitude"],
            [fn("MAX", col("magnitude")), "maxMagnitude"]
        ]
    });
    console.log(range)
}


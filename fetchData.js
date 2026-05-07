import moment from "moment";
import { addData } from "./sqlite.js";

const url = "https://earthquake.usgs.gov/fdsnws/event/1/query";

async function getData(starttime) {
    const params = {
        format: "geojson",
        starttime,
        endtime: moment(starttime).add(1, "year").format("YYYY-MM-DD"),
        minmagnitude: 4
    };

    try {
        const response = await fetch(url + "?" + new URLSearchParams(params));
        const data = await response.json();
        const datas = data.features.map((feature) => ({
            magnitude: feature.properties.mag,
            time: moment(new Date(feature.properties.time)).format("YYYY-MM-DD HH:mm:ss"),
            latitude: feature.geometry.coordinates[1],
            longitude: feature.geometry.coordinates[0]
        }));

        await Promise.all(datas.map((value) => addData(value)));
    } catch (error) {
        console.log(params, error.message);
    }
}

async function getAllData() {
    let starttime = "1901-01-01";
    const endtime = moment("2025-01-01").format("YYYY-MM-DD");

    while (starttime < endtime) {
        await getData(starttime);
        starttime = moment(starttime).add(1, "year").format("YYYY-MM-DD");
    }
}

getAllData();

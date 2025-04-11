let url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
let moment = require("moment");
let fs = require("fs");
const { addData } = require("./sqlite");

async function getData(starttime) {
    let params = {
        "format": "geojson",
        starttime,
        "endtime": moment(starttime).add(1, 'year').format("YYYY-MM-DD"), // moment 加一天
        "minmagnitude": 4,
    }
    let json = {};
    try {
        var d = await fetch(url + "?" + new URLSearchParams(params));
        // .then(response => response.json())
        var data = await d.json()
    } catch (error) {
        console.log(params, error.message);
    }

    let features = data.features;
    let datas = [];
    let earthquakes = features.map(feature => {
        datas.push({
            "magnitude": feature.properties.mag,
            "time": moment(new Date(feature.properties.time)).format("YYYY-MM-DD HH:mm:ss"),
            "latitude": feature.geometry.coordinates[1],
            "longitude": feature.geometry.coordinates[0]
        })
    });
    let proms = []

    datas.forEach(v=>{
        let p = addData(v)
        proms.push(p)
    })
    await Promise.all(proms)
}
async function getAllData() {
    let starttime = "1901-01-01"
    let endtime = moment("2025-01-01").format("YYYY-MM-DD")
    while (starttime < endtime) {
        await getData(starttime)
        starttime = moment(starttime).add(1, 'year').format("YYYY-MM-DD")
    }
}
getAllData();
import { DatabaseSync } from "node:sqlite";

export const db = new DatabaseSync("./database.sqlite");

db.exec(`
CREATE TABLE IF NOT EXISTS Earthquakes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    magnitude FLOAT NOT NULL,
    time VARCHAR(255) NOT NULL UNIQUE,
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL
)
`);

const insertEarthquake = db.prepare(`
INSERT INTO Earthquakes (magnitude, time, latitude, longitude)
VALUES (?, ?, ?, ?)
`);

const selectEarthquakesByMagnitude = db.prepare(`
SELECT time, latitude, longitude, magnitude
FROM Earthquakes
WHERE magnitude >= ?
ORDER BY time ASC
`);

const selectRange = db.prepare(`
SELECT
    MIN(time) AS minTime,
    MAX(time) AS maxTime,
    MIN(latitude) AS minLatitude,
    MAX(latitude) AS maxLatitude,
    MIN(longitude) AS minLongitude,
    MAX(longitude) AS maxLongitude,
    MIN(magnitude) AS minMagnitude,
    MAX(magnitude) AS maxMagnitude
FROM Earthquakes
`);

const selectEarthquakeCountsByYear = db.prepare(`
SELECT strftime('%Y', time) AS year, COUNT(*) AS count
FROM Earthquakes
WHERE magnitude >= ?
GROUP BY year
ORDER BY year ASC
`);

export function addData({ magnitude, time, latitude, longitude }) {
    try {
        const result = insertEarthquake.run(magnitude, time, latitude, longitude);
        console.log({ id: result.lastInsertRowid, magnitude, time, latitude, longitude });
        return result;
    } catch (error) {
        console.error("Error adding data:", error.message);
        return null;
    }
}

export function queryEarthquakes(minMagnitude = 6) {
    return selectEarthquakesByMagnitude.all(minMagnitude);
}

export function queryRange() {
    return selectRange.get();
}

export function countEarthquakesByYear(minMagnitude = 5) {
    try {
        const results = selectEarthquakeCountsByYear.all(minMagnitude);

        results.forEach(({ year, count }) => {
            console.log(`Year: ${year}, Earthquakes: ${count}`);
        });

        return results;
    } catch (error) {
        console.error("Error counting earthquakes by year:", error.message);
        return [];
    }
}

export function closeDb() {
    db.close();
}

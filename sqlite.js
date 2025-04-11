const { Sequelize, DataTypes, Op } = require("sequelize");

// 连接到内存中的 SQLite 数据库（临时）
// const sequelize = new Sequelize("sqlite::memory:");

// 或连接到文件型 SQLite 数据库（持久化）
const sequelize = new Sequelize({
    dialect: "sqlite",
    storage: "./database.sqlite", // 数据库文件路径
    logging: false // 关闭 SQL 日志（可选）
});

let Earthquake = sequelize.define("Earthquake", {
    id: {
        type: DataTypes.INTEGER,
        primaryKey: true,
        autoIncrement: true
    },
    magnitude: {
        type: DataTypes.FLOAT,
        allowNull: false
    },
    time: {
        type: DataTypes.STRING,
        allowNull: false
    },
    latitude: {
        type: Sequelize.FLOAT,
        allowNull: false
    },
    longitude: {
        type: Sequelize.FLOAT,
        allowNull: false
    }
}, { timestamps: false,indexes:[{fields:["time"],unique: true}] }); 
// Earthquake.sync();
async function addData(params) {
    try {
        let res = await Earthquake.create(params)
        console.log(res.toJSON())  
    } catch (error) {
        console.error("Error adding data:", error.message);
    }

}

async function countEarthquakesByYear(minMagnitude = 5) {
    try {
        const results = await Earthquake.findAll({
            attributes: [
                [Sequelize.fn("strftime", "%Y", Sequelize.col("time")), "year"],
                [Sequelize.fn("COUNT", "*"), "count"]
            ],
            where: {
                magnitude: {
                    [Op.gte]: minMagnitude // 过滤 5 级以上地震
                }
            },
            group: ["year"],
            order: [["year", "ASC"]]
        });

        results.forEach(result => {
            const { year, count } = result.dataValues;
            console.log(`Year: ${year}, Earthquakes: ${count}`);
        });

        return results.map(result => result.dataValues);
    } catch (error) {
        console.error("Error counting earthquakes by year:", error.message);
    }
}

// countEarthquakesByYear()
module.exports = {
    addData
}
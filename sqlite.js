const { Sequelize, DataTypes } = require("sequelize");

// 连接到内存中的 SQLite 数据库（临时）
// const sequelize = new Sequelize("sqlite::memory:");

// 或连接到文件型 SQLite 数据库（持久化）
const sequelize = new Sequelize({
    dialect: "sqlite",
    storage: "./database.sqlite", // 数据库文件路径
    logging: false // 关闭 SQL 日志（可选）
});

let earth = sequelize.define("Earthquake", {
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

async function addData(params) {
    // await earth.sync({ alter: { drop: false } })
    // 
    // {
    //     magnitude: 5.0,
    //     time: "2024-01-01",
    //     latitude: 35.0,
    //     longitude: 135.0
    // }
    try {
        let res = await earth.create(params)
        console.log(res.toJSON())  
    } catch (error) {
        console.error("Error adding data:", error.message);
    }


}
module.exports = {
    addData
}
// addData();
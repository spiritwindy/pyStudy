import opensim as osim

# 加载人体模型
model = osim.Model('path_to_your_model.osim')
model.setUseVisualizer(True)

# 设置关节运动范围和约束
hip_joint = model.getJointSet().get('hip')
hip_joint.getCoordinate().setRange([-1.57, 1.57])  # 假设仰卧起坐的髋关节范围是 -90 到 90 度

# 定义肌肉控制策略
controller = osim.PrescribedController()
controller.addActuator(model.getActuators().get('hip_flexor'))
controller.prescribeControlForActuator('hip_flexor', osim.Constant(0.5))  # 设置髋屈肌激活

# 添加控制器到模型
model.addController(controller)

# 创建仿真管理器并运行仿真
state = model.initSystem()
manager = osim.Manager(model)
manager.setInitialTime(0)
manager.setFinalTime(2.0)  # 设置仿真时间
manager.integrate(state)

# 可视化并分析结果
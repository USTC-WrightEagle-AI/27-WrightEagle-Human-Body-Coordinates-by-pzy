# 头部相机人体静态姿态识别

本目录迁移自
[`USTC-WrightEagle-AI/27-WrightEagle.Human-static-posture-recognition-by-pzy`](https://github.com/USTC-WrightEagle-AI/27-WrightEagle.Human-static-posture-recognition-by-pzy)。
原项目使用 ROS 1 Noetic 和另一台 D455；当前版本已适配本机 ROS 2 Humble
及头部 D455。

系统使用 YOLOv8-Pose 提取 COCO17 骨架，通过头部 D455 对齐深度反投影为
三维骨架，再由 24 帧因果 TCN 识别站立、坐下、蹲、躺和弯腰。当前配置处理
画面中的所有人员。

当前版本还使用 ByteTrack 为每个人维持 `person_id`。人脸子进程每次只对
整帧运行一次 InsightFace，再根据 YOLOv8-Pose 的耳朵、眼睛、鼻子和肩膀
关键点为每个身体生成虚拟脸框。真实脸框与虚拟脸框使用 IoU、修正 IoU、
中心距离和尺度一致性评分，通过 Hungarian 全局一对一分配后绑定到
ByteTrack ID。姓名仍需要在同一 ID 上多帧确认；深度位置、姓名和姿态
共用同一个 ByteTrack ID。

人脸与身体关联门限位于 `config/pose_3d_config.yaml` 的 `face_recognition`
段：`keypoint_confidence` 是生成虚拟脸框的关键点置信度门限，
`min_assignment_iou` 和 `min_assignment_score` 分别是候选对的最低 IoU 和综合
分数。降低门限会提高远距离召回率，但会增加多人串名风险。

## 本机相机配置

- 相机：头部 Intel RealSense D455
- 序列号：`260722304986`
- 流配置：`1280x720@30`
- RGB：`/head/camera/color/image_raw`
- 对齐深度：`/head/camera/aligned_depth_to_color/image_raw`
- 内参：`/head/camera/color/camera_info`
- 标定快照：`../calibration/head_d455_hd/`

联合识别的人脸检测器输入为 `1280`，与独立录入和识别一致；人体姿态模型输入仍为 `640`。

节点启动时会核对相机参数服务中的序列号和 RGB/Depth profile，并将实时
`CameraInfo` 与高清流对应的工厂内参比较。任何设备、分辨率、帧率或内参
不匹配都会拒绝启动。

`camera.down_axis` 是 2026-10-16 在机器人静止时连续采集 5 秒 D455 IMU
数据，再通过设备内部 IMU 到彩色光学系外参转换得到：
`[-0.004230, 0.997762, 0.066737]`。它相对理想图像向下方向
`[0, 1, 0]` 偏离约 `3.83 deg`，主要是相机俯仰。该重力轴不使用
`handeye_candidate_v3.json` 中未验收的左臂外参；头部相机安装姿态变化后
必须重新测量。

## 运行

先启动机器人 bringup，确认头部 RGB 和对齐深度话题存在，然后运行：

```bash
cd /home/rpp/Robocup/pzy-cs
./posture_recognition/scripts/start_3d.sh
```

默认使用本机已有的 `foundationpose` Conda 环境。无可视化图像输出时：

```bash
./posture_recognition/scripts/start_3d.sh --no-image
```

跳过部分输入帧可降低 GPU 占用，例如每两帧处理一帧：

```bash
./posture_recognition/scripts/start_3d.sh --frame-skip 1
```

## ROS 2 输出

- `/pose_classifier_3d/results`：完整姿态、ID、姓名和坐标 JSON
- `/person_identity_locations`：精简的 ID、姓名和坐标 JSON
- `/pose_classifier_3d/image`：`sensor_msgs/msg/Image`，骨架与姿态可视化

查看结果：

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /pose_classifier_3d/results
ros2 topic echo /person_identity_locations
./posture_recognition/scripts/view_image.sh
```

精简输出示例：

```json
{
  "camera_coordinate_frame": "head_camera_color_optical_frame",
  "camera_coordinate_convention": "+X right, +Y down, +Z forward",
  "left_base_transform_validated": false,
  "persons": [
    {
      "person_id": 1,
      "name": "张三",
      "name_known": true,
      "face_score": 0.71,
      "camera_xyz_m": [0.10, 0.42, 2.27],
      "left_base_xyz_m": [2.39, 0.21, -0.20],
      "left_base_transform_validated": false
    }
  ]
}
```

`camera_xyz_m` 是可信主输出，坐标约定为相机光学系 `+X` 向右、`+Y` 向下、
`+Z` 向前。`left_base_xyz_m` 来自昨天的候选手眼外参；由于该外参未通过旋转
验收，其 `left_base_transform_validated` 固定为 `false`，不能直接用于机械臂
抓取或安全相关动作。

姓名来自当前项目的 `data/faces.npz`。只有人脸清晰可见并连续两次通过阈值后，
才会把姓名绑定到 ByteTrack ID；否则输出 `Unknown`。录入或更新人员仍使用：

```bash
cd /home/rpp/Robocup/pzy-cs
/home/rpp/miniconda3/envs/robot-face/bin/python remember_person.py 张三 --replace
```

不要在当前 Conda `base` 环境中直接执行 `ros2 run rqt_image_view ...`：该环境的
`python3` 是 3.14，而 ROS 2 Humble 的 `rclpy` 扩展是为系统 Python 3.10 编译的。
查看脚本会显式使用 `/usr/bin/python3`，不受当前 Conda 环境影响。等价命令为：

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 /opt/ros/humble/lib/rqt_image_view/rqt_image_view \
  /pose_classifier_3d/image
```

## 模型

- `models/yolov8n-pose.pt`：本机 CUDA PyTorch/Ultralytics 使用的姿态检测模型
- `models/pose_tcn_ntu_fukinect.pth`：上游 3D TCN 分类权重
- `models/yolov8n-pose.engine`：保留的上游 Jetson TensorRT 文件，本机不使用

校验模型：

```bash
cd /home/rpp/Robocup/pzy-cs/posture_recognition/models
sha256sum -c SHA256SUMS
```

## 测试

```bash
cd /home/rpp/Robocup/pzy-cs
source /opt/ros/humble/setup.bash
export PYTHONPATH=/home/rpp/Robocup/pzy-cs/posture_recognition/src:$PYTHONPATH
/home/rpp/miniconda3/envs/foundationpose/bin/python \
  -m unittest discover -s posture_recognition/tests -v
```

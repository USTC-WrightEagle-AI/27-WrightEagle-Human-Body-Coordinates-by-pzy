# 机器人本地人脸识别

人体静态姿态识别已经迁移到本机头部 D455，使用说明见
[`posture_recognition/README.md`](posture_recognition/README.md)。启动命令：

```bash
./posture_recognition/scripts/start_3d.sh
```

项目提供两个独立脚本：

- `remember_person.py`：采集面前一人的多帧人脸，使用 RetinaFace 检测和 ArcFace-R50 提取特征，并把平均特征保存到本地。
- `recognize_person.py`：实时检测画面中的人脸，与本地特征库进行余弦相似度匹配，输出姓名或 `Unknown`。

预训练模型采用 InsightFace `buffalo_l` 模型包，其中 `det_10g.onnx` 是 RetinaFace 检测器，`w600k_r50.onnx` 是 ArcFace-R50 识别模型。

## 环境安装

推荐单独建立 Python 3.10 环境，避免影响机器人已有的 GraspNet 环境。先加载 Conda 的 shell 配置：

```bash
cd /home/rpp/pzy-cs
source /home/rpp/miniconda3/etc/profile.d/conda.sh
conda create -n robot-face --override-channels -c conda-forge python=3.10 pip -y
conda activate robot-face
python --version
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

`python --version` 必须显示 Python 3.10。也可以完全绕过环境激活状态，用下面的命令安装：

```bash
conda run -n robot-face python -m pip install -r requirements.txt
```

不要使用 `base` 环境的 Python 3.14；NumPy 1.x 和部分 InsightFace 依赖不支持该版本。

首次运行时 InsightFace 会下载 `buffalo_l` 到 `~/.insightface/models/buffalo_l`。下载完成后可以离线运行。

确认 ONNX Runtime 能看到 GPU：

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

输出应包含 `CUDAExecutionProvider`。如果没有，可以暂时传入 `--provider cpu`，但速度会明显降低。

## 记住一个人

默认使用头部 D455（SDK 序列号 `260722304986`），按 2026-10-15 标定配置读取 `640×480@15` 彩色和对齐深度图。画面中出现多人时，只录入深度中位数最小的最近人脸：

```bash
python remember_person.py 张三
```

脚本默认收集 20 个合格样本。人员已经存在时，需要明确覆盖：

```bash
python remember_person.py 张三 --replace
```

特征默认保存在 `data/faces.npz`，其中不包含原始照片。

## 删除已录入人员

先查看当前人脸库：

```bash
python manage_faces.py list
```

删除指定人员。默认会要求确认：

```bash
python manage_faces.py delete "张三"
```

确认后也可以使用：

```bash
python manage_faces.py delete "张三" --yes
```

该操作只删除指定人员的特征，不会删除其他人员。

最近两个人的深度差默认必须达到 0.15 米，否则本帧暂停采集。可按现场情况修改：

```bash
python remember_person.py 张三 --min-depth-gap 0.20 --replace
```

要临时关闭深度过滤并恢复“画面只能有一人”的模式：

```bash
python remember_person.py 张三 --no-depth-filter --replace
```

## 识别人脸

```bash
python recognize_person.py
```

默认余弦相似度阈值为 `0.45`。误认较多时提高到 `0.50` 或 `0.55`；经常把本人判为未知时可以在验证后略微降低：

```bash
python recognize_person.py --threshold 0.50
```

无显示器运行，并输出便于机器人程序读取的 JSON Lines：

```bash
python recognize_person.py --headless --json
```

输出示例：

```json
{"timestamp": 1788144600.0, "name": "张三", "known": true, "score": 0.7123}
```

## 头部相机及标定配置

录入和识别均默认 `--camera head`，默认订阅现有 `/head/camera` ROS 图像流，并向相机节点核对 SDK 序列号，不依赖易变的 `/dev/videoX` 编号。默认配置在 `calibration/head_d455/`，从 `/home/rpp/caliberate/data/handeye/head_d455/` 保存了 2026-10-15 的独立快照：

- `session.json`：设备序列号和 `640×480@15` 流配置。
- `intrinsics.json`：工厂 RGB 内参、畸变参数、深度参数和设备 RGB/Depth 外参。
- `handeye_candidate_v3.json`：头部相机到左臂基座的候选外参，保留来源记录。

启动时检查内参摘要、ROS 相机序列号、流配置及实时 RGB 内参。焦距允许 0.5% 的小幅变化（与姿态节点一致，兼容机器人实测约 0.300% 的焦距变化），主点和畸变系数须与保存值一致；不会把旧焦距强制写入设备。不允许在头部模式中随意修改分辨率或帧率；更改后应导出对应流的新内参及 session，再使用 `--calibration-dir`。ROS 相机驱动或 SDK 使用设备工厂参数完成深度对齐，识别模型仍接收原始 BGR 彩色图，不做额外的图像坐标或机械臂坐标变换。

v3 平均残差约为 `3.96 mm / 0.544°`，旋转残差尚未通过标定检查。人脸身份匹配只需要图像和特征，不依赖该机械臂外参；此项目不输出左臂基座系人脸坐标，也不会控制机械臂。

使用现有环境直接启动，无需重新安装：

```bash
cd /home/rpp/pzy-cs
/home/rpp/miniconda3/envs/robot-face/bin/python recognize_person.py
```

默认 `--head-backend ros`，保留机器人现有 ROS 相机节点运行：

- 彩色图：`/head/camera/color/image_raw`
- 工厂内参反馈：`/head/camera/color/camera_info`
- 对齐深度：`/head/camera/aligned_depth_to_color/image_raw`

图像订阅由加载 `/opt/ros/humble/setup.bash` 的系统 Python 子进程负责，人脸推理继续使用 `robot-face` 环境，继承当前 ROS 域和通信配置（`ROS_LOCALHOST_ONLY` 未指定时默认 `1`）。录入时按消息时间戳配对彩色与深度（最大间隔 50 ms），拒绝缺失或过期数据；识别仅订阅彩色和内参。不需要停止原 ROS 相机节点。

若未运行 ROS 相机节点，可显式使用 SDK 直接打开相机；此时先退出其他占用头部 D455 的程序：

```bash
python recognize_person.py --head-backend sdk
python remember_person.py 张三 --head-backend sdk
```

默认 ROS 模式不会因连接失败而自动切换相机或输入方式。关闭深度筛选时仍读取同一头部彩色图，只是不使用深度筛选人脸。

若要显式使用其他摄像头，保留原设备输入方式，例如腕部 D405：

```bash
python recognize_person.py --camera /dev/video16 --width 1280 --height 720 --fps 30
python remember_person.py 张三 --camera /dev/video16 --realsense-serial 260322273941
```

已有 `data/faces.npz` 保持不变，可以直接用于头部相机匹配；如现场效果不佳，可用头部相机手动重新录入，覆盖时须加 `--replace`。

## 参数建议

- 人脸距离：默认要求检测框短边至少达到 80 像素；还要拉远时可传入 `--min-face-size 70`，不建议低于 64 像素。
- 录入：正脸为主，缓慢向左右转头，避免强逆光。
- 阈值：必须用实际人员和摄像头数据校准，不要只依赖默认值。
- 数据：`data/faces.npz` 属于生物特征数据，应限制读取权限并避免上传到代码仓库。

InsightFace 提供的预训练模型有其单独的使用许可，默认面向非商业研究用途。商业部署前需要确认模型授权。

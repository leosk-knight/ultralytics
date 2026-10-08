# Project model YAML files

把项目自己的模型 YAML 放在这里，并使用明确的版本名，例如：

```text
yolo26n_custom_v1.yaml
yolo26n_custom_attention_v2.yaml
```

推荐的演进顺序：

1. 只使用现有模块时，复制对应官方 YAML 后仅修改层连接或类别数。
2. 新增模块时，把 Python 模块放在 `ultralytics/nn/modules/`，再在 YAML 中引用它。
3. 需要训练逻辑变化时，新增派生 Trainer，而不是复制整个官方训练器。

不要把 `*.pt` 权重放入这个目录；权重由实验制品库管理，模型 YAML 和对应 Git commit
一起保存。

## YOLOv8m 检测 + 人体姿态双头

`yolov8m-dual.yaml` 共享 YOLOv8m backbone 和 neck，末端 `DualDetectPose` 分别输出检测框与人体框/关键点。通过项目入口训练、验证和预测：

```python
from custom.dual_task import DualTaskYOLO

model = DualTaskYOLO("custom/configs/models/yolov8m-dual.yaml")
model.train(data="path/to/data.yaml", pretrained="yolov8m.pt", epochs=100)

model = DualTaskYOLO("path/to/best.pt")
metrics = model.val(data="path/to/data.yaml")
print(metrics.pose.map, metrics.det.box.map)
for result in model.predict("path/to/image.jpg"):
    print(result.boxes)                 # 检测头
    print(result.pose.boxes)            # 姿态头的人体框
    print(result.pose.keypoints)        # 姿态头的关键点
```

数据集 YAML 需有 `names` 和 `kpt_shape: [17, 3]`。标签按 YOLO Pose 格式逐行保存：
`class x_center y_center width height` 后跟 17 组 `x y visibility`。所有类别都写满关键点列；
非人体目标的关键点全部填 `0 0 0`。模型 YAML 中的 `person_class` 指向检测类别里的人体编号。
姿态损失只使用该类别，检测损失使用全部类别。验证会分别计算两个头的指标。

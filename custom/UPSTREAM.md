# Upstream synchronization record

本项目基于官方 Ultralytics 仓库维护：

- Upstream: <https://github.com/ultralytics/ultralytics.git>
- Tracking branch: `upstream/main`
- Local base at scaffold creation: `b5f6c7024`

- Latest synchronization: `7866d1217` (`upstream/main`)
- Sync date: `2026-09-09`


## Synchronization checklist

每次同步完成后更新本文件：

```text
Upstream commit: 7866d1217 (Bump to 8.4.145 for login failure exit status)
Sync date: 2026-09-09
Validation: `yolo checks`; `python3 -m compileall -q ultralytics`; CPU COCO8 one-epoch smoke test passed
Notes: Rebased the maintenance scaffold onto upstream/main; no conflicts.
```

## Local extension inventory

- `custom/`: project maintenance files and local configuration templates.
- `ultralytics/nn/modules/`: custom PyTorch modules, when added.
- `custom/configs/models/`: project model YAML files.
- `custom/configs/datasets/`: dataset configuration templates.

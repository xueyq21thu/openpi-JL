# Finetune Readme

## Debugging

### 1. Debug lerobot
你需要去修改lerobot源码：

> /workspace/openpi-JL/.venv/lib/python3.11/site-packages/lerobot/common/datasets/lerobot_dataset.py

`LeRobotDatasetMetadata`的以下部分：

```python
class LeRobotDatasetMetadata:
    def __init__(self, repo_id: str, *, local_files_only: bool = False):
        self.repo_id = repo_id
        self.local_files_only = local_files_only

        # ✅ 这里加判断，防止对本地数据集下载
        if not local_files_only:
            self.pull_from_repo(allow_patterns="meta/")
```

以及该文件下面`LeRobotDataset`的以下部分：

```python
def download_episodes(self, download_videos: bool = False):
    ...
    if not self.local_files_only:
        self.pull_from_repo(allow_patterns=files, ignore_patterns=ignore_patterns)
```

以及确保`lerobot_dataset.py`的`__init__.py`文件中有以下代码：

```python
self.local_files_only = local_files_only
```

# CAL Readme

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

## Action in LIBERO

LIBERO基于robosuite框架构建，后者是一个用于机器人操作任务的模拟环境。在robosuite中，动作（action）通常由一个NumPy数组表示，其维度和含义取决于所使用的机器人和控制模式。

在您的示例中，动作数组为

```
[-0.29855708 -0.01579233 -0.40058467 -0.00119008 -0.02454117  0.00938399 -0.97967208]
```


这与LIBERO代码中的`LIBERO_DUMMY_ACTION`一致：


```python
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
```


这表明动作数组包含7个元素。在robosuite中，通常的动作格式包括：

1. **前三个元素**：对应于末端执行器（例如机械臂末端）的笛卡尔空间位置控制（X, Y, Z）。

2. **第四至第六个元素**：对应于末端执行器的旋转，可能以轴角（axis-angle）或四元数的形式表示。

3. **第七个元素**：对应于夹爪的开合程度，通常为一个标量值，表示夹爪的开合状态。

因此，您的动作数组的各个分量可能表示：

- **前三个值**：末端执行器在X、Y、Z方向上的位置偏移。

- **第四至第六个值**：末端执行器的旋转角度，可能采用轴角表示法。

- **第七个值**：夹爪的开合状态，通常为-1表示关闭，1表示打开。

请注意，具体的动作格式可能因机器人模型和控制模式的不同而有所变化。建议查阅您使用的特定环境和机器人模型的文档，以获取最准确的信息。 
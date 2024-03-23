# 动物王国竞赛代码库

该代码仓库主要用于参加ICME(2024)多模型推理和分析大赛其中基于动物王国数据集的动物呀行为识别赛道(#6)。我们基于MMAction2框架利用该数据集训练了Video Swin Transformer模型，在动物行为识别实现任务上取得了较高的准确度。

## 数据准备

本仓库的所有训练/验证都只需要视频数据集进行参与。可以通过官方途径或者[百度网盘]()进行下载，将其解压放置于`data/AnimalKingdom/dataset/video`下。

![image-20240323223609938](https://raw.githubusercontent.com/ChengAoShen/Image-Hosting/main/images/image-20240323223609938.png)

> 如果需要特定的路径可以在配置文件之中进行修改

## 模型下载与加载

最佳的模型可以从[此处]()进行下载，其中包含不同Loss函数训练的一系列模型，文件名字按照`overallmap_headmap_middlemap_tailmap`。下载后建议将其放置于`data/model`路径下。

加载特定的模型通过修改配置文件中`load_from`变量进行实现。更多关于配置文件的部分将在模型训练/测试部分介绍。

## 模型训练/测试

模型的训练与测试主要使用MMAction自带的工具以及配置文件。所有的配置文件位于`mmaction2/work_dirs`路径下，主要提供swin-large模型在不同loss函数下的实现。更多关于配置文件的内容可以见[mmaction2官方文档](https://mmaction2.readthedocs.io/zh-cn/latest/user_guides/train_test.html)。

假设当前路径为`mmaction2`，不同的指令如下所示

* 单卡模型训练

  ```bash
  python tools/train.py ${CONFIG_FILE}
  ```

  例如：`python tools/train.py work_dirs/swin-large_FocalLoss.py`

* 多卡模型训练：

  ```bash
  bash tools/dist_train.sh ${CONFIG} ${GPUS}
  ```

  例如：`bash tools/dist_train.sh work_dirs/swin-large_FocalLoss.py 8`

* 单卡测试：

  ```bash
  python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
  ```

  例如：`python tools/test.py work_dirs/swin-large_FocalLoss.py ../data/model/5263_6577_5861_4773.pth`

* 多卡测试：

  ```bash
  bash tools/dist_test.sh ${CONFIG} ${CHECKPOINT} ${GPUS}
  ```

  例如：`bash tools/dist_test.sh work_dirs/swin-large_FocalLoss.py ../data/model/5263_6577_5861_4773.pth 8`

> 更多配置文件的修改可以根据需求进行。

## License

Apache-2.0 license

# Animal Kingdom Competition Code Repository

> [Chinese Document](https://github.com/ChengAoShen/Animal_Kingdom/blob/main/README_zh-CN.md)

This code repository is mainly used for participating in the ICME (2024) multimodal inference and analysis competition, specifically the animal behavior recognition track (#6) based on the Animal Kingdom dataset. We have trained the Video Swin Transformer model using this dataset based on the MMAction2 framework, achieving high accuracy in the task of animal behavior recognition.

## Data Preparation

All training/validation in this repository only requires the video dataset. It can be downloaded through official channels or [Baidu Netdisk](https://pan.baidu.com/s/1mQQJwVIRWrnyeqjdwqJXoQ?pwd=f1D5), and then extracted to `data/AnimalKingdom/dataset/video`.

![image-20240323223609938](https://raw.githubusercontent.com/ChengAoShen/Image-Hosting/main/images/image-20240323223609938.png)

> If a specific path is needed, it can be modified in the configuration file.

## Model Download and Loading

The best model can be downloaded from [here](), which includes a series of models trained with different loss functions, with the file names formatted as `overallmap_headmap_middlemap_tailmap`. After downloading, it is recommended to place them in the `data/model` path.

Loading a specific model can be achieved by modifying the `load_from` variable in the configuration file. More details about the configuration file will be introduced in the model training/testing section.

## Model Training/Testing

Model training and testing mainly use the tools and configuration files that come with MMAction. All configuration files are located in the `mmaction2/work_dirs` path, mainly providing implementations of the Swin-large model under different loss functions. More details about the content of the configuration files can be seen in the [mmaction2 official documentation](https://mmaction2.readthedocs.io/en/latest/user_guides/train_test.html).

Assuming the current path is `mmaction2`, different commands are as follows:

* Single card model training:

  ```bash
  python tools/train.py ${CONFIG_FILE}
  ```

  For example: `python tools/train.py work_dirs/swin-large_FocalLoss.py`

* Multi-card model training:

  ```bash
  bash tools/dist_train.sh ${CONFIG} ${GPUS}
  ```

  For example: `bash tools/dist_train.sh work_dirs/swin-large_FocalLoss.py 8`

* Single card testing:

  ```bash
  python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
  ```

  For example: `python tools/test.py work_dirs/swin-large_FocalLoss.py ../data/model/5263_6577_5861_4773.pth`

* Multi-card testing:

  ```bash
  bash tools/dist_test.sh ${CONFIG} ${CHECKPOINT} ${GPUS}
  ```

  For example: `bash tools/dist_test.sh work_dirs/swin-large_FocalLoss.py ../data/model/5263_6577_5861_4773.pth 8`

> Modifications to more configuration files can be made as needed.

## License

Apache-2.0 license
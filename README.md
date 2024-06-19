
本项目为计算机视觉-2024春结课作业

## Requirements

运行以下命令安装依赖：

```bash
pip install -r requirements.txt
```

## Pretrained Model

运行前，在 `pretrained` 文件夹下放置预训练模型，可以从以下命令下载：

```
wget https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl -O  pretrained/R50.pkl 
```

## Train

参数被存储在configs文件夹下，推荐使用`config/final.yaml`配置文件，运行以下命令即可训练实验中最后的配置：

```bash
CUDA_VISIBLE_DEVICES=0, python main.py --cfg config/final.yaml
```

如果想运行RCTW预训练，使用 `config/pretrain_on_text.yaml` 配置文件，运行
```bash
CUDA_VISIBLE_DEVICES=0, python main.py --cfg config/pretrain_on_text.yaml
```

## Eval


可以运行 `eval.ipynb` 来互动地测试模型的结果，默认的测试集均为BJTU，或者使用以下命令查看结果：

```bash
CUDA_VISIBLE_DEVICES=0, python eval.py --weight_dir result/final_r50_b4_s3000_rc3
```

如果想查看单张效果，可以运行以下命令：

```bash
CUDA_VISIBLE_DEVICES=0, python predict_img.py --weight_dir result/final_r50_b4_s3000_rc3 --image_path BJTU_washed/test/fhy-2nj3in4df.jpg
```


## 准备数据集

这部分的许多内容在实验时在Notebook中完成，参数被写死，这里只是简单的说明。

### BJTU

数据集文件夹被重命名为BJTU_washed，应当放在项目根目录下。数据集的文件夹结构应组织如下，

```
BJTU_washed/
    test/
        fhyxxxx.jpg
        ...
    train/
        fhyxxxx.jpg
        ...
```

首先去除重复文件，注意提前修改其中指定的目录并运行

```bash
python data_wash.py
```

然后运行以下命令生成COCO数据文件：

```bash
python build_coco.py
```

或者也可以从链接下载后将 test.json 和 train.json 放到 BJTU_washed 目录下


### RCTW

RCTW数据集可从 [RCTW](https://rctw.vlrlab.net/) 下载。下载后放到项目根目录下，结构应为

```
RCTW_17/
    raw/
        train_images/
            image_0.jpg
            ...
    rctw_train.json
```

COCO格式的json可以自己生成，也可以直接从我提供的地址下载

## 其他文件

doc/ 下存放了结果数据的csv文件
处理数据集时去除的文件列表存放在 doc/train_dup.txt 和 doc/test_dup.txt 中



目录结构
===

```bash
.
├── # data/ #
├── layers.py
├── load_data.py
├── loss.py
├── network.py
├── README.md
├── result/
├── run_mlp.py
├── run.py
├── scripts
│   └── run_all.sh
├── solve_net.py
└── utils.py

# 原始数据并未包含于此目录中, 如需复现实验，请先创建目录./data/，并将将数据文件移至./data/
```

重要文件说明
===

run.py
---

对run_mlp.py的改进，以便从命令行传入参数，方便实验，并进行绘图。

使用方法 `./run.py {-name [NAME]} {-config [CONFIG]} {-arch [ARCH]} {-loss [LOSS]}`

参数说明

+ name: 实验名称。默认为“default”
+ config: 修改的超参数，如“learning_rate:0.1 weight_decay:0.001”将修改learning_rate为0.1，修改weight_decay为0.001，其余沿用默认参数。默认为“”
+ arch: 模型结构，如“Lin-784-10 Sigm”将构建 (Linear, Sigmoid)的神经网络模型。默认为“Lin-784-10 Relu”
+ loss: 损失函数，euclidean或softmax。默认为“euclidean”

使用范例可参见./scripts/run_all.sh

scripts/run_all.sh
---

里面设计了所有的实验，如需复现实验，请先创建目录./data/，并将将数据文件移至./data/，然后在终端运行 `./scripts/run_all.sh`，结果将存于./result/

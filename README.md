# TS_HDQN

===========================

## 环境依赖
python  3.7.9\
torch   1.7.0+cpu


## 部署步骤
1. 安装python包\
    pip install -r piplist.txt
3. 生产仿真数据\
    python gen_data.py
3. H-DQN训练\
    python h_dqn.py
4. 算法性能对比\
    python main.py


## 目录结构描述

```
├── Readme.md                   // help
├── piplist.txt                 // python依赖包列表
├── data                        
│   ├── fig                     // 算法对比图    
│   ├── model                   // 训练完成的网络
│   └── result                  // 实验数据
├── main.py                     // 算法性能对比           
├── h_dqn.py                    // Hierarchy DQN
├── dqn.py                      // Deep Q Network
├── model_nn.py                 // 神经网络模型
├── environment.py              // 模拟环境
├── baseline.py                 // 对比算法：随机、贪婪
├── gen_data.py                 // 生成实验数据
├── configuration.py            // 配置数据规模
└── tools.py                    // 包括经验回放ERM、画图plot工具
```

## 内容更新
1. 批标准化 or 输入数据归一化


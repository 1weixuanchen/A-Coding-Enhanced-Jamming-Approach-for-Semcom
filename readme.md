Can Knowledge Improve Security? A Coding-Enhanced Jamming Approach for Semantic Communication
===

Accepted by 2025 IEEE JSAC
---

This repository contains the official implementation of the paper: “Can Knowledge Improve Security? A Coding-Enhanced Jamming Approach for Semantic Communication.” The method introduces a coding-enhanced jamming mechanism that leverages shared private knowledge between transmitter and receiver to generate private digital codebooks, enabling secure digital semantic communication even when the eavesdropper has the same channel SNR as the legitimate user.

A-coding-enhanced-jamming-approach-for-semcom/

│── basic_module.py/   # Code for the basic modules

│── cka.py/            # nHSIC implementation code

│── dataset.py/        # Data set loading

│── eval.py/           # evaluation codes

│── main.py/           # Build a complete training/testing process

│── pretrain_net.py/   # Pre-trained module loading

│── requirements.txt/   

│── scm_main_net.py/   # network architecture

│── train.py/           

│── test.py/           

└── README.md          # specification

1. Create environment

conda create -n cej_semcom python=3.8 -y

conda activate cej_semcom

---

2. Install dependencies

pip install -r requirements.txt

---

3. Training Pipeline

python3 main.py --alpha_1 1.0 --alpha_2 1.5 --alpha_3 0.01 --channel_use 128 --snr_train_leg 10 --mode train

---

4. Testing Pipeline

python3 main.py --channel_use 128 --snr_test_leg 10 --mode test

---

If you used our code or methods in your research, please cite:

@article{chen2025knowledge, 

title={Can Knowledge Improve Security? A Coding-Enhanced Jamming Approach for Semantic Communication}, 

author={Weixuan Chen and Qianqian Yang and Shuo Shao and Zhiguo Shi and Jiming Chen and Xuemin Shen}, 

journal={arXiv:2504.16960v4 [cs.IT]}, 

year={Sep. 2025} }

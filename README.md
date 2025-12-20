# [AAAI 2026] SPJFNet: Self-Mining Prior-Guided Joint Frequency Enhancement for Ultra-Efficient Dark Image Restoration

- *Tongshun Zhang, Pingping Liu, Zijian Zhang, Qiuzhan Zhou*
- *College of Computer Science and Technology, Jilin University*
- *Key Laboratory of Symbolic Computation and Knowledge Engineering of Ministry of Education*
- *College of Communication Engineering, Jilin University*

## 🔥 News

- [11/2025] SPJFNet accepted to AAAI 2026.

## 1. Abstract
Current dark image restoration methods suffer from severe efficiency bottlenecks, primarily stemming from: computational burden and error correction costs associated with reliance on external priors (manual or cross-modal); redundant operations in complex multi-stage enhancement pipelines; and indiscriminate processing across frequency components in frequency-domain methods, leading to excessive global computational demands. To address these challenges, we propose an Efficient Self-Mining Prior-Guided Joint Frequency Enhancement Network (SPJFNet). Specifically, we first introduce a Self-Mining Guidance Module (SMGM) that generates lightweight endogenous guidance directly from the network, eliminating dependence on external priors and thereby bypassing error correction overhead while improving inference speed. Second, through meticulous analysis of different frequency domain characteristics, we reconstruct and compress multi-level operation chains into a single efficient operation via lossless wavelet decomposition and joint Fourier-based advantageous frequency enhancement, significantly reducing parameters. Building upon this foundation, we propose a Dual-Frequency Guidance Framework (DFGF) that strategically deploys specialized high/low frequency branches (wavelet-domain high-frequency enhancement and Fourier-domain low-frequency restoration), decoupling frequency processing to substantially reduce computational complexity. Rigorous evaluation across multiple benchmarks demonstrates that SPJFNet not only surpasses state-of-the-art performance but also achieves significant efficiency improvements, substantially reducing model complexity and computational overhead.

## Motivation

![Motivation](https://github.com/user-attachments/assets/87fd721f-05c9-47c3-a107-14b72d750539)

## SPJFNet

![Over-all-Architecture](https://github.com/user-attachments/assets/791eac49-acda-48b7-aa26-fd368426868c)


## 2. Prepare Dataset

LOL-v1 [Baidu Disk](https://pan.baidu.com/share/init?surl=ZAC9TWR-YeuLIkWs3L7z4g&pwd=cyh2) (code: cyh2), [Google Drive](https://drive.google.com/file/d/1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H/view?usp=sharing). From [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer).

LSRW [Baidu Disk](https://pan.baidu.com/s/1XHWQAS0ZNrnCyZ-bq7MKvA) (code:wmrr). From [R2RNet](https://github.com/JianghaiSCU/R2RNet).


### 3. Train
```
python train.py -opt ./options/train/huawei.yml
```

### 4. Test
```
python test.py -opt ./options/test/huawei.yml
```

### 5. Pre-trained Models and Outputs
✨ [Google Drive](Updating ... ).


## Citation Information

This repo is based on [DMFourLLIE](https://github.com/bywlzts/DMFourLLIE).

If you find the project useful, please cite:  

```bibtex  
@article{zhang2025spjfnet,
  title={SPJFNet: Self-Mining Prior-Guided Joint Frequency Enhancement for Ultra-Efficient Dark Image Restoration},
  author={Zhang, Tongshun and Liu, Pingling and Zhang, Zijian and Zhou, Qiuzhan},
  journal={arXiv preprint arXiv:2508.04041},
  year={2025}
}
@inproceedings{zhang2024dmfourllie,  
    title={DMFourLLIE: Dual-Stage and Multi-Branch Fourier Network for Low-Light Image Enhancement},  
    author={Zhang, Tongshun and Liu, Pingping and Zhao, Ming and Lv, Haotian},  
    booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},  
    pages={7434--7443},  
    year={2024}  
}  


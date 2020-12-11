# BicycleGAN
*This project serves as the final project of CIS680 Fall 2020: Advanced Topics in Machine Perception at University of Pennsylvania.*

**Authors: Kun Huang, Zhihao Ruan**

This project explores BicycleGAN implementation with modified ResNet generator from CycleGAN, PatchGAN discriminator, and ResNet encoder. The training process is built on the dataset `edges2shoes`.

The model trained on our PC reaches an **FID score of 76.822**, and **LPIPS of 0.23297.** For more information on FID score & LPIPS metric, visit:
- FID score: https://github.com/mseitzer/pytorch-fid
- LPIPS metric: https://github.com/richzhang/PerceptualSimilarity

## Quick Start
### Training
To train the network on `edges2shoes` dataset, at the project root folder run:
```bash
sh scripts/download_and_train.sh
```

### Inference & Evaluation
To run inferences as well as generate quantitative evaluation on the trained model, run
```bash
sh scripts/infer.sh
```



## Reference
> **Toward Multimodal Image-to-Image Translation.**
> 
> [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/),
>  [Richard Zhang](https://richzhang.github.io/), [Deepak Pathak](http://people.eecs.berkeley.edu/~pathak/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/), [Oliver Wang](http://www.oliverwang.info/), [Eli Shechtman](https://research.adobe.com/person/eli-shechtman/).  
>  UC Berkeley and Adobe Research  
> In Neural Information Processing Systems, 2017. 

GitHub link: https://github.com/junyanz/BicycleGAN


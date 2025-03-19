# MPPE
- **Title: Multi-modal Prompts with Primitives Enhancement for Compositional Zero-Shot Learning （MPPE）**
- **Authors: Yutang Jin, Shiming Chen, Tianle Tong, Weiping Ding, Yisong Wang.**
- **Institutes: Guizhou University, Huazhong University of Science and Technology, Nantong University.**

This paper is currently under review at TCSVT. Once accepted, we will release the model code and weights, and further improve this open-source project.

## Overview
<p align="center">
  <img src="images/Fig3.png" />
</p>
The overview of the proposed MPPE model. MPPE is based on CLIP, with two primitive prompts and one compositional prompt, which are totally learnable. Additionally, a visual prompt named Alpha branch is introduced to help focus on regions of interest, by binary masks from SAM model with corresponding object label, especially, <font color="red">$\dagger$</font> denotes the label is only used in the training phase. Another main part of our work is the primitives enhancement module(termed as PE), primitive semantic features from text encoder are fed into PE to generate enhanced compositional semantic features, as an extra compositional prediction branch. Specifically, the primitives enhancement module is based on cross-attention, object and attribute semantic features $t^{o}$ and $t^{a}$ are fed into and as query and key, value respectively, followed with nonlinear transformations, finally the enhanced compositional semantic features $\hat{T_{c}}$ is generated. Additionally, a learnable coefficient $\lambda$ is included for controlling the weight of enhanced compositional semantic features.

## Acknowledgement
This project mostly references [[Troika]](https://github.com/bighuang624/Troika) and [[DFSP]](https://github.com/Forest-art/DFSP), and I once again express my sincere gratitude to the authors of these two papers!

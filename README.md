# Neural Style Transfer: Paper Explanations
This repository documents my study of foundational papers in neural style transfer. Below are detailed explanations of key concepts from each paper.

## 📄 1. Perceptual Losses for Real-Time Style Transfer and Super-Resolution  
Johnson et al. (2016) 
[[Paper Link]](https://arxiv.org/abs/1603.08155)

### Loss Functions

Traditional image transformation networks often rely on per-pixel losses, which fail to capture perceptual and semantic differences between images. Perceptual losses address this limitation by leveraging high-level features from pre-trained classification networks. Johnson et al. propose two key perceptual loss functions:

#### Feature Reconstruction Loss
The feature reconstruction loss penalizes differences in content between the output and target images by measuring the Euclidean distance between their feature representations:

$$\ell^{\phi,j}_{feat}(\hat{y}, y) = \frac{1}{C_j H_j W_j} \|\phi_j(\hat{y}) - \phi_j(y)\|^2_2$$

where $\phi_j(x)$ represents the activations at the $j$-th layer of the pre-trained network $\phi$ for image $x$, with dimensions $C_j \times H_j \times W_j$. This loss encourages the output image to be perceptually similar to the target without forcing exact pixel matching.

#### Style Reconstruction Loss
The style reconstruction loss captures differences in textures, colors, and patterns by comparing the Gram matrices of feature representations:

$$\ell^{\phi,j}_{style}(\hat{y}, y) = \|G^{\phi}_j(\hat{y}) - G^{\phi}_j(y)\|^2_F$$

where the Gram matrix $G^{\phi}_j(x)$ is defined as:

$$G^{\phi}_j(x)_{c,c'} = \frac{1}{C_j H_j W_j} \sum_{h=1}^{H_j} \sum_{w=1}^{W_j} \phi_j(x)_{h,w,c} \phi_j(x)_{h,w,c'}$$

This matrix represents the correlations between different feature channels, effectively capturing style information independent of spatial structure.

## 📄 2. Texture Networks: Feed-forward Synthesis of Textures and Stylized Images  
Ulyanov et al. (2016)  
[[Paper Link]](https://arxiv.org/abs/1603.03417)  

### Texture Synthesis Loss  
The method trains a **feed-forward generator network** \( \mathbf{g}(\mathbf{z}; \theta) \) to synthesize textures by minimizing a texture loss derived from Gram matrix statistics. Given a reference texture \( \mathbf{x}_0 \), the loss matches feature correlations across layers of a fixed VGG descriptor network:  

$$  
\mathcal{L}_T(\mathbf{x}; \mathbf{x}_0) = \sum_{l \in L_T} \| G^l(\mathbf{x}) - G^l(\mathbf{x}_0) \|_2^2  
$$  

where \( G^l(\mathbf{x})_{ij} = \langle F^l_i(\mathbf{x}), F^l_j(\mathbf{x}) \rangle \) computes the Gram matrix for layer \( l \), and \( F^l_i(\mathbf{x}) \) denotes the \( i \)-th feature map at layer \( l \). This enforces texture similarity by preserving channel-wise correlations.  

### Style Transfer Objective  
For stylization, the generator \( \mathbf{g}(\mathbf{y}, \mathbf{z}; \theta) \) takes a content image \( \mathbf{y} \) and noise \( \mathbf{z} \), and optimizes a combined loss:  

$$  
\theta_{\mathbf{x}_0} = \arg\min_\theta \, \mathbb{E}_{\mathbf{z}, \mathbf{y}} \left[ \mathcal{L}_T(\mathbf{g}(\mathbf{y}, \mathbf{z}; \theta), \mathbf{x}_0) + \alpha \mathcal{L}_C(\mathbf{g}(\mathbf{y}, \mathbf{z}; \theta), \mathbf{y}) \right]  
$$  

The **content loss** \( \mathcal{L}_C \) matches high-level features at layer \( l \in L_C \):  

$$  
\mathcal{L}_C(\mathbf{x}, \mathbf{y}) = \sum_{l \in L_C} \sum_{i=1}^{N_l} \| F^l_i(\mathbf{x}) - F^l_i(\mathbf{y}) \|_2^2  
$$  

where \( \alpha \) balances style and content fidelity.  

### Multi-Scale Architecture  
The generator uses a **multi-scale convolutional design** with noise inputs \( \{\mathbf{z}_i\} \) at different resolutions. Each scale processes noise through convolutional blocks, upsampling layers, and channel-wise concatenation. The architecture is fully convolutional, enabling arbitrary output sizes. Batch normalization stabilizes training, and lightweight parameters (~65K) ensure real-time synthesis.  

---  
**Key Innovation**: Shifts computational cost to training, enabling **single-pass generation** at ~500x speedup over optimization-based methods like Gatys et al., while maintaining perceptual quality.

# Fast Style Transfer PyTorch

A PyTorch implementation of fast neural style transfer based on the papers:
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)
- [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)

This implementation uses **Instance Normalization** instead of Batch Normalization for better stylization results.

## Features

- Fast style transfer using feedforward networks
- Instance normalization for improved quality
- GPU acceleration support
- Easy training and evaluation scripts
- Clean PyTorch implementation

## Requirements

```bash
pip install torch torchvision pillow numpy
```

## Usage

### Training

Train a new style transfer model:

```bash
python train.py train --dataset /path/to/coco/dataset \
                     --style-image /path/to/style/image.jpg \
                     --save-model-dir ./models \
                     --epochs 2 \
                     --batch-size 4 \
                     --cuda 1
```

### Stylization

Apply a trained model to stylize an image:

```bash
python train.py eval --content-image /path/to/content/image.jpg \
                    --model ./models/trained_model.pth \
                    --output-image ./output.jpg \
                    --cuda 1
```

## Model Architecture

- **Transform Network**: Encoder-decoder architecture with residual blocks
- **Loss Network**: Pre-trained VGG16 for perceptual losses
- **Instance Normalization**: Applied after each convolutional layer
- **Residual Connections**: 5 residual blocks in the bottleneck

## Training Details

- **Content Loss**: Computed using VGG16 relu2_2 features
- **Style Loss**: Computed using Gram matrices from multiple VGG16 layers
- **Optimizer**: Adam with learning rate 1e-3
- **Image Size**: 256x256 during training

## Dataset

For training, you'll need:
- **MS COCO dataset** for content images
- **Style images** (paintings/artistic images)

Download MS COCO:
```bash
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip
```

## Results

The trained models can stylize images in real-time while preserving content structure and applying artistic style effectively.

## References

- Original TensorFlow implementation by Logan Engstrom
- Johnson et al. "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"
- Ulyanov et al. "Instance Normalization: The Missing Ingredient for Fast Stylization"

## License

This implementation is for research and educational purposes.

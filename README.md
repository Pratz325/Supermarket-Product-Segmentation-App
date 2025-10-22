# Supermarket Product Instance Segmentation

Mask R-CNN based instance segmentation system for automated supermarket checkout.

## Features
- Upload product images
- Real-time instance segmentation
- Confidence threshold adjustment
- Detailed detection statistics
- Processing time metrics

## Usage
1. Upload an image
2. Adjust confidence threshold
3. View segmentation results

## Model
- Architecture: Mask R-CNN (ResNet50-FPN)
- Classes: 61 (60 products + background)
- mAP@0.5:0.95: 44.7%
- Best for: Large products (>50k px²)

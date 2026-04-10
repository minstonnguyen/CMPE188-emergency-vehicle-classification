# ML Pipeline Architecture

This document shows the current MVP architecture for the emergency vehicle classification project.

## End-to-End Flow

```mermaid
flowchart TD
    A[Raw image datasets\nRoboflow / Images.cv / synthetic smoke-test data] --> B[Prepared dataset folders\ntrain / val / test]
    B --> C[Data loader\nImageFolder + transforms]
    C --> D[Training loop\nforward pass + loss + backprop]
    D --> E[Model checkpoint\nbest_model.pt]
    D --> F[Validation metrics\nloss / accuracy]
    E --> G[Test evaluation]
    G --> H[outputs/test_metrics.json]
    E --> I[Single-image inference]
    I --> J[Predicted class + confidence]
```

## Training Architecture

```mermaid
flowchart LR
    A[configs/baseline.yaml] --> B[cli_train.py]
    B --> C[data.py]
    B --> D[model.py\nSmallCNN]
    B --> E[engine.py]
    C --> E
    D --> E
    E --> F[Best checkpoint]
    E --> G[training_history.json]
    E --> H[test_metrics.json]
```

## Inference Architecture

```mermaid
flowchart LR
    A[Input image] --> B[load_image]
    C[best_model.pt] --> D[SmallCNN]
    B --> D
    D --> E[Softmax scores]
    E --> F[Predicted label]
    E --> G[Confidence]
```

## Component Summary

- `configs/baseline.yaml`: runtime configuration for paths and training hyperparameters
- `src/emergency_vehicle_classifier/data.py`: dataset loading and image preprocessing
- `src/emergency_vehicle_classifier/model.py`: CNN model definition
- `src/emergency_vehicle_classifier/engine.py`: training and evaluation logic
- `src/emergency_vehicle_classifier/cli_train.py`: training entrypoint
- `src/emergency_vehicle_classifier/cli_infer.py`: inference entrypoint
- `outputs/`: saved metrics for experiments
- `models/`: saved trained checkpoints

## Current MVP Boundaries

- Input is image classification, not object detection
- Inference is single-image prediction, not video tracking
- Metrics are written to JSON for reporting and reproducibility
- The same structure can be extended later to real datasets and richer models

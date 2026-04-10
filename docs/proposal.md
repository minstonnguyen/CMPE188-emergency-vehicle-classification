# Proposal Submission

## Title

Classification model for identifying law enforcement and/or emergency vehicles to alert users to drive with care

## Proposed Application

Project Option 1

## Problem Statement

We propose a computer vision system that identifies law enforcement and emergency vehicles from images and recorded video frames. The practical goal is to warn users when these vehicles are present so they can drive more carefully. To keep the first milestone realistic, we will begin with image classification and then extend to video inference if the model performs well enough.

## Task Definition

Primary task:
- Binary image classification of `emergency_vehicle` vs `non_emergency_vehicle`

Stretch goal:
- Multi-class classification of `police`, `ambulance`, `fire truck`, and `non_emergency`
- Frame-by-frame video inference on recorded footage

This keeps the project well-scoped while still allowing a stronger final demo if time permits.

## Possible Datasets

We plan to use public data from the following sources:

- Roboflow police vehicle datasets: https://universe.roboflow.com/search?q=class%3Apolice+car
- Police Car Image Classification Dataset on Images.cv: https://images.cv/dataset/police-car-image-classification-dataset

We will also collect negative examples such as normal cars, taxis, security vehicles, and other visually similar vehicles so the model learns to distinguish emergency vehicles from lookalikes.

## Features of the Datasets

The datasets contain RGB vehicle images captured in different environments and viewpoints. Useful visual features include:

- Vehicle color patterns and body shape
- Police or emergency markings and logos
- Emergency light bars
- Reflective decals and distinctive striping
- Scene context such as roads, parking lots, and urban settings

These images include variation in angle, lighting, and background, which is important for training a model that generalizes beyond a single location.

## Proposed Method

We will start with a convolutional neural network baseline for image classification. The first phase will focus on:

1. Cleaning and merging public datasets
2. Standardizing labels across sources
3. Training a baseline CNN on labeled images
4. Measuring accuracy, precision, recall, and confusion matrix
5. Testing on unseen images

If the baseline is strong enough, we will extend the project to recorded video by running the classifier on extracted frames or by adapting the approach to object detection.

## Final Demo

Our final demo will show the model identifying law enforcement or emergency vehicles in new images and, if feasible, in recorded video. The demo will likely include:

- Sample predictions on unseen test images
- Confidence scores for each prediction
- A short recorded video showing the model running on frames from real-world footage

If video performance is not reliable enough, we will present a strong image-based demo and discuss how the system can be extended to video.

## Risks and Challenges

- Police and emergency vehicles vary across cities and departments
- Lighting, motion blur, and occlusion may reduce accuracy
- Positive examples may be easier to find than strong negative examples
- Similar vehicles such as security cars may cause false positives

We will address this by using more diverse data and including visually similar non-emergency vehicles in training.

## Reference List

- Roboflow Police Vehicle datasets
- Police Car Image Classification Dataset on Images.cv

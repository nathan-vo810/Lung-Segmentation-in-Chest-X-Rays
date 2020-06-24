# Lung Segmentation in Chest X-ray images using Deep Neural Networks
Thesis Project for Computer Science Department at Vietnamese - German University

## Abstract
Lung disease is a serious problem nowadays. The number of patients with lung disease is getting higher every year. Unfortunately, not everyone has a chance to be treated with advanced technology. This is because of the cost, and experts, doctors shortage. The current state of computer-aided diagnosis systems is also not effective and efficient enough to support doctors in reducing workload. Hence, it is necessary and urgent to develop such system that can support the work of doctors and as the result help more patients to get in contact with those innovations. This paper aims to build such system dueling specifically with lung disease. The system can learn to segment the lung region in the X-ray image by applying deep learning - a fast-growing technology in recent years.

With the help of this diagnosis system, the work required to investigate fur- ther lung-related problems will probably decrease. Although there is still a lot of work ahead to complete the system, the proposed approach has shown promis- ing results and points out that deep learning is going to assist medical tremen- dously.

## Introduction
### Motivation
Lung cancer is a critical problem nowadays as the world is getting more polluted. According to statistical report early this year from Cancer Prevention Research Institute, in Vietnam lung cancer is the number one reason of death in man and ranked second in woman. Every year, there are 22,000 new cases and 19,500 patients die. There will be approximately over 34,000 new lung cancer cases in 2020 . These numbers have pointed out that the problem of lung cancer is get- ting worse every year. The need for diagnosing lung diseases are higher than ever. While there are more and more patients, the number of experts is limited.

Despite its high in demand in medical practice, chest X-ray images screening and diagnose is still a challenge. In a recent survey, van Ginneken et al claim that there isn’t any computer-aid detection (CAD) system that is good enough to handle chest radiographs diagnose. To effectively evaluating lung diseases, the segmentation of lung region is the first step in the process. Traditional techniques for lung segmentation can be categorized as active shapes, rule-based methods, pixel classification, and various combination thereof. They proposed that pixel level classification would provide a highly accurate segmentation of the lung field.

In recent years, the significant growth of machine learning, and especially deep learning has helped to improve many aspects of computer vision tasks in various fields such as in automotive like self-driving car, or in marketing for analyzing of images using object recognition or face recognition techniques, or the analyzing of video for recognizing relevant scenes. From LeNet for simple hand-written digits classification, to AlexNet and VGGNet that are able to classification million of images in the ImageNet Challenge with the top-5 error 26%, and to the more complex deep network model GoogLeNet that can achieve a top-5 error of only 6.7%. The health-care industry has also benefited a lot from this. With the help of machine learning algorithms, computer-aided diagnosis (CAD) systems have become more essential than ever. They also become part of routine hospital work for detection of critical diseases. Those networks have helps scientist to figure out solutions to apply into semantic segmentation problem. Specifically, UNet is a Fully Convolutional Network that has been proposed to apply into biomedical applications such as cell tracking.

### Objective
This project is conducted to apply those state-of-the-art methods into a CAD system particularly for lung diagnosis. To be more precise, the system will apply deep learning method, convolutional neural networks, to tackle the challenge. The purpose of the project is to help segment the lung region from chest X-ray images and thus, support both doctors and patients in preventing, curing, and fighting lung diseases. Moreover, this project also encourages me to explore in the field of machine learning, deep learning, and their applications in real world.

## Proposed Method
To address the problem of lung segmentation in chest X-ray images, I proposed two methods:
 - The first method is designing a convolutional neural network model.
 - The second method is applying U-Net, a state-of-the-art model.

### HybridNet
The problem of lung segmentation can be treated as a classification problem whether a pixel is lung or non-lung. Therefore, for this approach, I introduced HybridNet - a convolutional neural network model for classification that takes an extracted region from the original image called window as input and output the probability of that window is lung or non-lung. Moreover, inspired by earlier work on lung segmentation by combining different features, HybridNet also incorporate prior knowledge into the model to support the prediction.

HybridNet is a combination between (1) a convolutional neural network and (2) a traditional neural network. The convolutional neural network is in charge of extracting the features from the input image which is a small region inside the X-ray image. The traditional neural network enables a way to feed the prior knowledge of the region location into the network. Together, the two networks try to predict whether a region inside the original X-ray image belongs to the lung or non-lung.

### UNet
UNet is a state-of-the-art model for fast and precise segmentation of images. Its prior application is to track cell. UNet learns segmentation in an end-to-end setting. It is a type of encode-decode network, for every encode layer there will be an accordingly decode layer. The final layer is a pixel-wise classification layer.

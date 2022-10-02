## Inspiration
As the scope of applying machine learning is fast-growing nowadays, more and more cases are reported regarding the degradation of algorithm quality regarding different users. When this degradation frequently happens to certain social groups, ethical issues will occur. We believe that along with developing advanced and accurate algorithms, the ethical usage of AI helps make sure this technology is sustainable and applied in a fair way. 

We specifically targeted Facial Recognition (FR) algorithm, for its variety in use cases and inherit sensitivity of biometric information. 

## What it does
The project is to address one of the ethical issues with Facial Recognition -- the biased behavior of machine learning algorithms regarding different races. The primary objective is to reduce false positive rate of prediction for dataset of colored individuals while keeping the model accuracy before applying debiasing architecture.  

## How we built it
We first researched the reason of racial bias in FR. In summary, demographic distribution of users of FR technology is the primary cause of this. Then, we brainstormed to propose a model that requires no significant re-training of already developed state-of-the-art by merging the results of models with different performance metrics judging by the confidence score of each. Next, to simulate the mitigation of such issues, we trained a neural network on a image recognition dataset, CIFAR100 (due to time constraints) using Pytorch library to practice the model we developed. 

Dataset used: CIFAR100 (https://www.cs.toronto.edu/~kriz/cifar.html), CTK_CB

## Challenges we ran into
1. That the accuracy of the model is very hard to achieve within 24 hours. We chose to move from the high-level task of facial recognition to item recognition to reduce the training time of our model. 

2. Virtual outlier synthesis. This is a research topic for making machine learning models better learn unfamiliar distributions, we read it from a paper and needed to find ways to implement it ourselves

## Accomplishments that we're proud of
1.  We proposed a complete machine learning pipeline, beginning at data cleaning and ending at evaluating model outputs. 

2. We researched the prominent problem of algorithmic bias and generated a practical solution that can be used to resolve such an issue in even real-world settings.

## What we learned
1. That machine learning in practice needs to take "human-factors" into account. The design of AI architectures will be impacted by the evaluation of them in real-world applications.

2. That training of machine learning algorithms requires time and dedication. For this reason, we needed to always have a plan B in mind when working on this hackathon.

## What's next for Only Believe What You Know
1. We will generalize the idea of OOD into facial recognition area by testing with more comprehensive facial datasets and using alternative architectures
2. Fixing Gaussian Distribution Assumption
3. We will test system with adversarial examples

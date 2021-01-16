# NTU-SPML2020
NTU CSIE5436 Security and Privacy of Machine Learning(2020 FALL)

## Homework1 Gray-box Attack
### Requirement
You will create untargeted adversarial examples to attack models for the CIFAR-10 classification task. (https://www.cs.toronto.edu/~kriz/cifar.html). Your goal is to bring down the model accuracy as much as possible.

Five models will be chosen from this repository: https://github.com/osmr/imgclsmob.  (連結到外部網站。)Some preprocessing defenses may be used to improve the model robustness.

You are allowed to change each pixel of the input image up to epsilon=8 on the 0-255 pixel scale. Of course, each pixel after the perturbation still needs to be within 0 to 255 in order to be a valid image.

Your attack will be evaluated based on the accuracy on the evaluation set (download here), which consists of 100 images from CIFAR-10 (10 images of each class).

You can use any programming languages and packages. Please add a README.txt file to tell people how to run your code.

You need to write a report describing your methods.  You can talk about, for example,  why you choose certain (combination of) methods and any internal experiments that you did (e.g., accuracy on substitute models, or against popular defenses). Please write it using Latex with the NeurIPS conference template (https://nips.cc/Conferences/2020/PaperInformation/StyleFiles (連結到外部網站。)). Report length is at most 4 pages, excluding references (please cite the work that you used in this homework).

### hw1/hw1_atk.py
Generate adversarial examples from CIFAR10 testing set.
Attack method includes:<br>
(1) FGSM<br>
(2) I-FGSM<br>
(3) MI-FGSM<br>
(4) PGD<br>
(5) Ensemble Attack<br>
```bash
usage: hw1_atk.py [-h] [--bs BS] [--eps EPS] [--attack ATTACK] [--iter ITER] [--alpha ALPHA] [--mu MU] [--rand_start RAND_START]
                  [--gpu GPU] [--model MODEL] [--model_ckpt MODEL_CKPT] [--models_file MODELS_FILE] [--save_file SAVE_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --bs BS               batch size (default: 16)
  --eps EPS             Linf constraint for attack (default: 0.03137254901960784)
  --attack ATTACK       attacks mehtods now supported: {fgsm/mifgsm} (default: mifgsm)
  --iter ITER           IFGSM iter (default: 10)
  --alpha ALPHA         IFGSM alaph (default: 0.1)
  --mu MU               IFGSM mu (default: 1.0)
  --rand_start RAND_START
                        random initialize as PGD (default: False)
  --gpu GPU             gpu number (default: 0)
  --model MODEL         pytorchcv model (default: )
  --model_ckpt MODEL_CKPT
                        model's checkpoint (default: )
  --models_file MODELS_FILE
                        pytorchcv models file (ignore if model not empty) (default: )
  --save_file SAVE_FILE
                        adv images save path (default: )
```

## Homework2 Black-box Defense
### Requirement
In this homework, you need to train a robust model for CIFAR-10 that can defend the adversarial examples. That is, you need to submit the code of model architecture and the trained weight, then TA will use adversarial examples (up to epsilon=8 in the L_infinity norm) to attack your model. Good luck.

Write a report with at most 4 pages in NeurIPS format

Methods you tried
Why you choose certain methods in your submission
Experiments that you did
Findings or insights you gained

### hw2/adv_train.py
Provide (1) Standard Adversarial Training (2) Friendly Adversarial Training (3) Intermittent Adversarial Training on pytorchcv models.
```bash
usage: adv_train.py [-h] [--sep SEP] [--bs BS] [--lr LR] [--epoch EPOCH] [--friendly FRIENDLY] [--rand_start RAND_START]
                    [--eps EPS] [--iter ITER] [--alpha ALPHA] [--mu MU] [--gpu GPU] [--log_path LOG_PATH]
                    [--save_path SAVE_PATH]
                    model

positional arguments:
  model                 training model (pytorchcv)

optional arguments:
  -h, --help            show this help message and exit
  --sep SEP             epoch gap for regenerate adversarial examples (default: 1)
  --bs BS               batch size (default: 128)
  --lr LR               learning rate (default: 0.1)
  --epoch EPOCH         epoch (default: 120)
  --friendly FRIENDLY   wether using friendly setting (default: False)
  --rand_start RAND_START
                        random initialize start point as PGD attack (default: True)
  --eps EPS             Linf constraint of IFGSM *default(8/255) (default: 0.03137254901960784)
  --iter ITER           iter of IFGSM (default: 10)
  --alpha ALPHA         alaph of IFGSM (default: 0.2)
  --mu MU               mu of IFGSM (default: 0.0)
  --gpu GPU             GPU device (default: 0)
  --log_path LOG_PATH   log file save to {log_path}_{model} (default: log)
  --save_path SAVE_PATH
                        ckpt file save to {save_path}_{model} (default: ckpt)
```
### hw2/test.py
Test model performance on specified input images and labels(tensor stored by torch.save).<br>
Include following additional defense method:<br>
(1) JPEG Compression<br>
(2) Defense-GAN<br>
```bash
usage: test.py [-h] [--jpeg JPEG] [--quality QUALITY] [--dfgan DFGAN] [--init_num INIT_NUM] [--rec_iter REC_ITER]
               [--testnum TESTNUM] [--ckpt CKPT] [--imgs IMGS] [--labels LABELS] [--bs BS] [--gpu GPU]
               model

positional arguments:
  model                test model name (pytorchcv model)

optional arguments:
  -h, --help           show this help message and exit
  --jpeg JPEG          Apply JPEG Compression preprocessing or not (default: False)
  --quality QUALITY    JPEG config: compression quality(0-100). High quality means little compression (default: 100)
  --dfgan DFGAN        Apply Defense-GAN preprocessing or not (default: False)
  --init_num INIT_NUM  Defense-GAN config: init_num (default: 5)
  --rec_iter REC_ITER  Defense-GAN config: rec_iter (default: 200)
  --testnum TESTNUM    number of testing images (default: 10000)
  --ckpt CKPT          checkpoint path of model / default as pytorchcv pretrained weights (default: )
  --imgs IMGS          path to images data(tensor) (default: )
  --labels LABELS      path to labels data(tensor) (default: )
  --bs BS              training batch_size (default: 64)
  --gpu GPU            gpu number (default: 0)
```


## Final Project
Poisoning Attack on Defense GAN
https://github.com/jasper0314-huang/NTU-SPML2020/blob/main/FinalProject_report.pdf

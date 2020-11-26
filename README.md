# NTU-SPML2020
NTU CSIE5436 Security and Privacy of Machine Learning(2020 FALL)

## Homework1 Gray-box Attack
### hw1/hw1_atk.py
Generate adversarial examples from CIFAR10 testing set.
Attack method includes:<br>
(1) FGSM<br>
(2) I-FGSM<br>
(3) MI-FGSM<br>
(4) PGD<br>
(5) Ensemble Attack<br>
```bash
usage: hw1_atk.py [-h] [--bs BS] [--eps EPS] [--attack ATTACK] [--iter ITER]
                  [--alpha ALPHA] [--mu MU] [--rand_start RAND_START]
                  [--gpu GPU] [--model MODEL] [--model_ckpt MODEL_CKPT]
                  [--models_file MODELS_FILE] [--save_file SAVE_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --bs BS               batch size (default: 16)
  --eps EPS             Linf constraint for attack (default:
                        0.03137254901960784)
  --attack ATTACK       attacks mehtods now supported: {fgsm/mifgsm} (default:
                        mifgsm)
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
                        pytorchcv models file (ignore if model not empty)
                        (default: )
  --save_file SAVE_FILE
                        adv images save path (default:
                        /tmp2/b07501122/SPML/adv_examples/cifar_adv_imgs)
```

## Homework2 Black-box Defense
### hw2/adv_train.py
Provide (1) Standard Adversarial Training (2) Friendly Adversarial Training (3) Intermittent Adversarial Training on pytorchcv models.
```bash
usage: adv_train.py [-h] [--sep SEP] [--bs BS] [--lr LR] [--epoch EPOCH]
                    [--friendly FRIENDLY] [--rand_start RAND_START]
                    [--eps EPS] [--iter ITER] [--alpha ALPHA] [--mu MU]
                    [--gpu GPU] [--log_path LOG_PATH] [--save_path SAVE_PATH]
                    model

positional arguments:
  model                 training model (pytorchcv)

optional arguments:
  -h, --help            show this help message and exit
  --sep SEP             epoch gap for regenerate adversarial examples
                        (default: 1)
  --bs BS               batch size (default: 128)
  --lr LR               learning rate (default: 0.1)
  --epoch EPOCH         epoch (default: 120)
  --friendly FRIENDLY   wether using friendly setting (default: False)
  --rand_start RAND_START
                        random initialize start point as PGD attack (default:
                        True)
  --eps EPS             Linf constraint of IFGSM *default(8/255) (default:
                        0.03137254901960784)
  --iter ITER           iter of IFGSM (default: 10)
  --alpha ALPHA         alaph of IFGSM (default: 0.2)
  --mu MU               mu of IFGSM (default: 0.0)
  --gpu GPU             GPU device (default: 0)
  --log_path LOG_PATH   log file save to {log_path}_{model} (default: log)
  --save_path SAVE_PATH
                        ckpt file save to {save_path}_{model} (default: ckpt)
```
### hw2/test.py
test model performance on specified input images and labels(tensor stored by torch.save)
```bash
usage: test.py [-h] [--dfgan DFGAN] [--init_num INIT_NUM]
               [--rec_iter REC_ITER] [--ckpt CKPT] [--imgs IMGS]
               [--labels LABELS] [--bs BS] [--gpu GPU]
               model

positional arguments:
  model                test model name (pytorchcv model)

optional arguments:
  -h, --help           show this help message and exit
  --dfgan DFGAN        Apply Defense-GAN preprocessing or not (default: False)
  --init_num INIT_NUM  Defense-GAN config: init_num (default: 5)
  --rec_iter REC_ITER  Defense-GAN config: rec_iter (default: 200)
  --ckpt CKPT          checkpoint path of model / default as pytorchcv
                       pretrained weights (default: )
  --imgs IMGS          path to images data(tensor) (default:
                       /tmp2/b07501122/SPML/adv_examples/cifar_imgs)
  --labels LABELS      path to labels data(tensor) (default:
                       /tmp2/b07501122/SPML/adv_examples/cifar_labels)
  --bs BS              training batch_size (default: 64)
  --gpu GPU            gpu number (default: 0)
```


## Final Project

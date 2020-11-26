# NTU-SPML2020
NTU CSIE5436 Security and Privacy of Machine Learning(2020 FALL)

## Homework1 Gray-box Attack
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


## Final Project

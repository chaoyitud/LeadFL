LeadFL: Client Self-Defense against  Model Poisoning in Federated Learning
---
This repository contains the code for the paper "Client Self-Defense against Model Poisoning in Federated Learning".

## Abstract
Federated Learning is highly susceptible to backdoor and targeted attacks as participants can manipulate their data and models locally without any oversight on whether they follow the correct process. There are a number of server-side defenses that mitigate the attacks by modifying or rejecting local updates submitted by clients. However, we find that bursty adversarial patterns with a high variance in the number of malicious clients can circumvent the existing defenses. 

We propose a client-self defense, LeadFL, that is combined with existing server-side defenses to thwart backdoor and targeted attacks. The core idea of LeadFL is a novel regularization term in local model training such that the Hessian matrix of local gradients is nullified. We provide the convergence analysis of LeadFL and its robustness guarantee in terms of certified radius. Our empirical evaluation shows that LeadFL is able to mitigate bursty adversarial patterns for both iid and non-iid data distributions. It frequently reduces the backdoor accuracy from more than 75% for state-of-the-art defenses to less than 10% while its impact on the main task accuracy is always less than for other client-side defenses. 

## Requirements
Requirements are listed in `requirements.txt`. To install them, run `pip install -r requirements.txt`.

## Quick Start
```
python3 -m fltk single ./configs/temlate.yaml
```
In the template, the client-side defense is `LeadFL`, the server-side defense is `Bulyan`, the attack is 9-pixel pattern backdoor attacks,and the dataset is `FashionMNIST`. You can change the parameters in the template to run other experiments.
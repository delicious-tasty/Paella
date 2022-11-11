# Paella
Conditional text-to-image generation has seen countless recent improvements in terms of quality, diversity and fidelity. Nevertheless, most state-of-the-art models require numerous inference steps to produce faithful generations, resulting in performance bottlenecks for end-user applications. In this paper we introduce Paella, a novel text-to-image model requiring less than 10 steps to sample high-fidelity images, using a speed-optimized architecture allowing to sample a single image in less than 500 ms, while having 573M parameters. The model operates on a compressed & quantized latent space, it is conditioned on CLIP embeddings and uses an improved sampling function over previous works. Aside from text-conditional image generation, our model is able to do latent space interpolation and image manipulations such as inpainting, outpainting, and structural editing.
<br>
<br>
![cover-figure](https://user-images.githubusercontent.com/117442814/201255417-8c8ca261-00c6-4526-8fc1-4bbd65b1c9d8.png)

<hr>

## Sampling
For sampling you can just take a look at the [sampling.ipynb](epic_link.com) notebook. :sunglasses:

## Train your own Paella
The main file for training will be [paella.py](paella.py). You can adjust all [hyperparameters](paella.py) to your own needs. During training we use webdataset, but you are free to replace that with your own custom dataloader. Just change the line on 72 in [paella.py](paella.py) to point to your own dataloader. Make sure it returns a tuple of ```(images, captions)``` where ```images``` is a ```torch.Tensor``` of shape ```batch_size x channels x height x width``` and captions is a ```List``` of length ```batch_size```. Now decide if you want to finetune Paella or start a new training from scratch:
### From Scratch
```
python3 paella.py
```
### Finetune
If you want to finetune you first need to download the [latest checkpoint and it's optimizer state](epic_download_link.py), set the [finetune hyperparameter](paella.py) to ```True``` and create a folder ```models/<RUN_NAME>``` and move both checkpoints to this folder. After that you can also just run:
```
python3 paella.py
```

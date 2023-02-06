import json
import queue
import torch.multiprocessing as mp
from collections import OrderedDict
import os
import time
import torch
import pandas as pd
from itertools import product
import torchvision
import numpy as np
import open_clip
from open_clip import tokenizer
from rudalle import get_vae
from einops import rearrange
import tensorflow.compat.v1 as tf
from PIL import Image
from evaluator import Evaluator
from modules import DenoiseUNet


def chunk(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def save_images_npz(path, save_path):
    arr = []
    base_shape = None
    for item in os.listdir(path):
        if os.path.isfile(os.path.join(path, item)) and item.endswith(".jpg"):
            img = Image.open(os.path.join(path, item))
            img = np.array(img.resize((256,256), Image.ANTIALIAS))
            if base_shape is None:
                base_shape = img.shape
            try:
                if img.shape == base_shape:
                    arr.append(img)
            except Exception as e:
                print(e)
                continue
    arr = np.stack(arr)
    print(arr.shape)
    np.savez(save_path, arr)


def log(t, eps=1e-20):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1., dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


def sample(model, c, x=None, mask=None, T=12, size=(32, 32), starting_t=0, temp_range=[1.0, 1.0], typical_filtering=True, typical_mass=0.2, typical_min_tokens=1, classifier_free_scale=-1, renoise_steps=11, renoise_mode='start'):
    with torch.inference_mode():
        r_range = torch.linspace(0, 1, T+1)[:-1][:, None].expand(-1, c.size(0)).to(c.device)
        temperatures = torch.linspace(temp_range[0], temp_range[1], T)
        if x is None:
            x = torch.randint(0, model.num_labels, size=(c.size(0), *size), device=c.device)
        if renoise_mode == 'start':
            init_x = x.clone()
        for i in range(starting_t, T):
            if renoise_mode == 'prev':
                prev_x = x.clone()
            r, temp = r_range[i], temperatures[i]
            logits = model(x, c, r)
            if classifier_free_scale >= 0:
                logits_uncond = model(x, torch.zeros_like(c), r)
                logits = torch.lerp(logits_uncond, logits, classifier_free_scale)
            x = logits
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, x.size(1))
            if typical_filtering:
                x_flat_norm = torch.nn.functional.log_softmax(x_flat, dim=-1)
                x_flat_norm_p = torch.exp(x_flat_norm)
                entropy = -(x_flat_norm * x_flat_norm_p).nansum(-1, keepdim=True)

                c_flat_shifted = torch.abs((-x_flat_norm) - entropy)
                c_flat_sorted, x_flat_indices = torch.sort(c_flat_shifted, descending=False)
                x_flat_cumsum = x_flat.gather(-1, x_flat_indices).softmax(dim=-1).cumsum(dim=-1)

                last_ind = (x_flat_cumsum < typical_mass).sum(dim=-1)
                sorted_indices_to_remove = c_flat_sorted > c_flat_sorted.gather(1, last_ind.view(-1, 1))
                if typical_min_tokens > 1:
                    sorted_indices_to_remove[..., :typical_min_tokens] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, x_flat_indices, sorted_indices_to_remove)
                x_flat = x_flat.masked_fill(indices_to_remove, -float("Inf"))
            x_flat = torch.multinomial(x_flat.div(temp).softmax(-1), num_samples=1)[:, 0]
            # print(x_flat.shape)
            # x_flat = gumbel_sample(x_flat, temperature=temp)
            x = x_flat.view(x.size(0), *x.shape[2:])
            if i < renoise_steps:
                if renoise_mode == 'start':
                    x, _ = model.add_noise(x, r_range[i+1], random_x=init_x)
                elif renoise_mode == 'prev':
                    x, _ = model.add_noise(x, r_range[i+1], random_x=prev_x)
                else: # 'rand'
                    x, _ = model.add_noise(x, r_range[i+1])
    return x.detach()


def encode(vq, x):
    return vq.encode(x)[-1]


def decode(vq, z):
    return vq.decode_indices(z)


class DatasetWriter:
    def __init__(self, date, base_path="/home/data"):
        self.date = date
        self.base_path = base_path

    def saveimages(self, imgs, captions, **kwargs):
        try:
            for img, caption in zip(imgs, captions):
                caption = caption.replace(" ", "_").replace(".", "")
                path = os.path.join(self.base_path, caption + ".jpg")
                torchvision.utils.save_image(img, path, **kwargs)
        except Exception as e:
            print(e)

    def save(self, que):
        while True:
            try:
                data = que.get(True, 1)
            except (queue.Empty, FileNotFoundError):
                continue
            if data is None:
                print("Finished")
                return
            sampled, captions = data["payload"]
            self.saveimages(sampled, captions)


class Sample:
    def __init__(self, date, device, cfg_weight=5, steps=8, typical_filtering=True, batch_size=8, base_path="/home/data", dataset="coco", captions_path="cap.parquet"):
        self.date = date
        self.cfg_weight = cfg_weight
        self.steps = steps
        self.typical_filtering = typical_filtering
        self.dataset = dataset
        self.captions_path = captions_path
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.base_path = base_path
        self.path = os.path.join(base_path, f"{steps}_{cfg_weight}_{typical_filtering}")
        self.model, self.vqmodel, self.clip, self.clip_preprocess = self.load_models()
        self.setup()
        self.que = mp.Queue()
        mp.Process(target=DatasetWriter(date, base_path=self.path).save, args=(self.que,)).start()

    def setup(self):
        if self.dataset == "coco":
            self.captions = pd.read_parquet(self.captions_path)["caption"]
        elif self.dataset == "laion":
            self.captions = pd.read_parquet(self.captions_path)["caption"]
        else:
            raise ValueError
        num_sampled = len(os.listdir(self.base_path))
        self.captions = self.captions[num_sampled:]
        os.makedirs(self.path, exist_ok=True)
        with open(os.path.join(self.path, "log.json"), "w") as f:
            json.dump({
                "date": self.date,
                "cfg": self.cfg_weight,
                "steps": self.steps
            }, f)

    def load_models(self):
        # --- Paella MODEL ---
        model_path = f"./models/Paella_f8_8192/model_600000.pt"
        state_dict = torch.load(model_path, map_location=self.device)
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k[7:]  # remove `module.`
        #     new_state_dict[name] = v
        model = DenoiseUNet(num_labels=8192, c_clip=1024).to(self.device)
        model.load_state_dict(state_dict)
        model.eval().requires_grad_()
        # --- VQ MODEL ---
        vqmodel = get_vae().to(self.device)
        vqmodel.eval().requires_grad_(False)
        # --- CLIP MODEL ---
        clip_model, _, _ = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k', cache_dir="/fsx/mas/.cache")
        del clip_model.visual
        clip_model = clip_model.to(self.device).eval().requires_grad_(False)
        clip_preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                             std=(0.26862954, 0.26130258, 0.27577711)),
        ])
        return model, vqmodel, clip_model, clip_preprocess

    @torch.no_grad()
    def r_decode(self, img_seq, shape=(32, 32)):
        img_seq = img_seq.view(img_seq.shape[0], -1)
        one_hot_indices = torch.nn.functional.one_hot(img_seq, num_classes=self.vqmodel.num_tokens).float()
        z = (one_hot_indices @ self.vqmodel.model.quantize.embed.weight)
        z = rearrange(z, 'b (h w) c -> b c h w', h=shape[0], w=shape[1])
        img = self.vqmodel.model.decode(z)
        img = (img.clamp(-1., 1.) + 1) * 0.5
        return img

    def convert_dataset(self):
        """
        path: base_path + folder + tar_file
        """
        batch_size = len(self.captions) / 8
        latent_shape = (32, 32)
        for cap in np.array_split(self.captions, batch_size):
            cap = list(cap)
            s = time.time()
            # print(len(cap))
            text = tokenizer.tokenize(cap).to(self.device)
            with torch.inference_mode():
                with torch.autocast(device_type="cuda"):
                    clip_embeddings = self.clip.encode_text(text).float()

                    sampled = sample(self.model, clip_embeddings, T=self.steps, size=latent_shape, starting_t=0,
                                     temp_range=[1.0, 1.0],
                                     typical_filtering=self.typical_filtering, typical_mass=0.2, typical_min_tokens=1,
                                     classifier_free_scale=self.cfg_weight, renoise_steps=self.steps - 1)
                sampled = self.r_decode(sampled, latent_shape)
            data = {
                "payload": [sampled, cap]
            }
            self.que.put(data)
            # print(f"Sampled {len(cap)} in {time.time() - s} seconds.")


if __name__ == '__main__':
    mp.set_start_method('spawn')

    date = "f8_600k_no_ema"
    dataset = "coco"
    base_dir = "/fsx/mas/paella_unet/evaluation"
    ref_images = os.path.join(base_dir, f"{dataset}_30k.npz")
    ref_captions = os.path.join(base_dir, f"{dataset}_30k.parquet")
    base_path = os.path.join(base_dir, date)
    os.makedirs(base_path, exist_ok=True)
    devices = [0, 1, 2, 3, 4, 5, 6, 7]
    cfgs = [3, 4, 5]  # [3, 4, 5, 8]
    steps = [12]  # [6, 8, 10, 12]
    typical_filtering = [True, False]

    combinations = iter(list(product(steps, cfgs, typical_filtering)))
    # chunked_combinations = chunk(combinations, n=len(devices))

    try:
        while True:
            processes = []
            for proc_id in devices:
                steps, cfg_weight, typical_filtering = next(combinations)
                while os.path.exists(os.path.join(base_path, f"{steps}_{cfg_weight}_{typical_filtering}")):
                    print(os.path.join(base_path, f"{steps}_{cfg_weight}_{typical_filtering}") + " already done. skipping....")
                    steps, cfg_weight, typical_filtering = next(combinations)
                print(f"Starting sampling with steps={steps}, cfg_weight={cfg_weight}.")
                conv = Sample(date, proc_id, steps=steps, cfg_weight=cfg_weight, typical_filtering=typical_filtering, batch_size=8, base_path=base_path, dataset=dataset, captions_path=ref_captions)
                processes.append(mp.Process(target=conv.convert_dataset))
                processes[proc_id].start()
            for p in processes:
                p.join()
    except StopIteration:
        if len(processes) > 0:
            for p in processes:
                p.join()
    print("Finished sampling....")

    for run in os.listdir(base_path):
        stat_dict = {}
        run_path = os.path.join(base_path, run)
        batch_path = os.path.join(run_path, "batch.npz")
        print(f"Converting {run_path} to npz....")
        save_images_npz(run_path, batch_path)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        evaluator = Evaluator(tf.Session(config=config))

        evaluator.warmup()

        ref_acts = evaluator.read_activations(ref_images)
        ref_stats, ref_stats_spatial = evaluator.read_statistics(ref_images, ref_acts)

        sample_acts = evaluator.read_activations(batch_path)
        sample_stats, sample_stats_spatial = evaluator.read_statistics(batch_path, sample_acts)

        prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])
        stat_dict["inception_score"] = evaluator.compute_inception_score(sample_acts[0])
        stat_dict["fid"] = sample_stats.frechet_distance(ref_stats)
        stat_dict["sfid"] = sample_stats_spatial.frechet_distance(ref_stats_spatial)
        stat_dict["prec"], stat_dict["recall"] = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])
        print("---------------------------------------------------------")
        print(f"Metrics for {run_path}")
        print(stat_dict)
        print("---------------------------------------------------------")
        json.dump(stat_dict, open(os.path.join(run_path, "stat_dict.json"), "w"))
        os.remove(batch_path)



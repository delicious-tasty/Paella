import copy
import math
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import wandb
from torch import nn, optim
import torchvision
from tqdm import tqdm
import time
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from modules import DenoiseGIC, DenoiseUNet, EMA
from utils import get_dataloader
import open_clip
from open_clip import tokenizer
from rudalle import get_vae


def encode(vq, x):
    return vq.model.encode((2 * x - 1))[-1][-1]


def decode(vq, z):
    return vq.decode(z.view(z.shape[0], -1))


def sample(model, c, x=None, mask=None, T=12, size=(32, 32), starting_t=0, temp_range=[1.0, 1.0], typical_filtering=True, typical_mass=0.2, typical_min_tokens=1, classifier_free_scale=4, renoise_steps=11, renoise_mode='start'):
    with torch.inference_mode():
        r_range = torch.linspace(0, 1, T+1)[:-1][:, None].expand(-1, c.size(0)).to(c.device)
        temperatures = torch.linspace(temp_range[0], temp_range[1], T)
        preds = []
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
            x = x_flat.view(x.size(0), *x.shape[2:])
            if i < renoise_steps:
                if renoise_mode == 'start':
                    x, _ = model.add_noise(x, r_range[i+1], random_x=init_x)
                elif renoise_mode == 'prev':
                    x, _ = model.add_noise(x, r_range[i+1], random_x=prev_x)
                else: # 'rand'
                    x, _ = model.add_noise(x, r_range[i+1])
            preds.append(x.detach())
    return preds


def train(proc_id, args):
    if os.path.exists(f"results/{args.run_name}/log.pt"):
        resume = True
    else:
        resume = False
    if not proc_id and args.node_id == 0:
        if resume:
            wandb.init(project="DenoiseGIT", name=args.run_name, entity="wand-tech", config=vars(args))
        else:
            wandb.init(project="DenoiseGIT", name=args.run_name, entity="wand-tech", config=vars(args))
        print(f"Starting run '{args.run_name}'....")
        print(f"Batch Size check: {args.n_nodes * args.batch_size * args.accum_grad * len(args.devices)}")
    parallel = len(args.devices) > 1
    device = torch.device(proc_id)

    vqmodel = get_vae().to(device)
    vqmodel.eval().requires_grad_(False)

    if parallel:
        torch.cuda.set_device(proc_id)
        torch.backends.cudnn.benchmark = True
        dist.init_process_group(backend="nccl", init_method="file://dist_file",
                                world_size=args.n_nodes * len(args.devices),
                                rank=proc_id + len(args.devices) * args.node_id)
        torch.set_num_threads(6)

    if args.model == "GIC":
        print(f"Model: DenoiseGIC")
        model = DenoiseGIC(num_labels=args.num_codebook_vectors, layers=36, c_hidden=1280).to(device)
    elif args.model == "UNet":
        model = DenoiseUNet(num_labels=args.num_codebook_vectors, c_clip=1024).to(device)
    else:
        raise NotImplementedError()

    if not proc_id and args.node_id == 0:
        print(f"Number of Parameters: {sum([p.numel() for p in model.parameters()])}")

    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')
    del clip_model.visual
    clip_model = clip_model.to(device).eval().requires_grad_(False)

    lr = 3e-4
    dataset = get_dataloader(args)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if not proc_id and args.node_id == 0:
        wandb.watch(model)
        os.makedirs(f"results/{args.run_name}", exist_ok=True)
        os.makedirs(f"models/{args.run_name}", exist_ok=True)

    grad_accum_steps = args.accum_grad
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                              steps_per_epoch=math.ceil(1000 / grad_accum_steps),
                                              epochs=600, pct_start=30 / 300, div_factor=25,
                                              final_div_factor=1 / 25, anneal_strategy='linear')

    if resume:
        if not proc_id and args.node_id == 0:
            print("Loading last checkpoint....")
        logs = torch.load(f"results/{args.run_name}/log.pt")
        start_step = logs["step"] + 1
        losses = logs["losses"]
        accuracies = logs["accuracies"]
        total_loss, total_acc = losses[-1] * start_step, accuracies[-1] * start_step
        ema = EMA(0.995, step=start_step)
        ema_model = copy.deepcopy(model)
        model.load_state_dict(torch.load(f"models/{args.run_name}/model.pt", map_location=device))
        ema_model.load_state_dict(torch.load(f"models/{args.run_name}/ema_model.pt", map_location=device))
        ema_model.eval().requires_grad_(False)
        if not proc_id and args.node_id == 0:
            print("Loaded model and EMA model.")
        opt_state = torch.load(f"models/{args.run_name}/optim.pt", map_location=device)
        last_lr = opt_state["param_groups"][0]["lr"]
        with torch.no_grad():
            for _ in range(logs["step"]):
                scheduler.step()
        if not proc_id and args.node_id == 0:
            print(f"Initialized scheduler")
            print(f"Sanity check => Last-LR: {last_lr} == Current-LR: {optimizer.param_groups[0]['lr']} -> {last_lr == optimizer.param_groups[0]['lr']}")
        optimizer.load_state_dict(opt_state)
        del opt_state
    else:
        losses = []
        accuracies = []
        start_step, total_loss, total_acc = 0, 0, 0
        ema = EMA(0.995)
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    if parallel:
        model = DistributedDataParallel(model, device_ids=[device], output_device=device)

    pbar = tqdm(enumerate(dataset, start=start_step), total=args.total_steps, initial=start_step) if args.node_id == 0 and proc_id == 0 else enumerate(dataset, start=start_step)
    model.train()
    for step, (images, captions) in pbar:
        images = images.to(device)
        with torch.no_grad():
            image_indices = encode(vqmodel, images)
            r = torch.rand(images.size(0), device=device)
            noised_indices, mask = model.module.add_noise(image_indices, r)

            if np.random.rand() < 0.1:  # 10% of the times...
                text_embeddings = images.new_zeros(images.size(0), 1024)
            else:
                text_tokens = tokenizer.tokenize(captions)
                text_tokens = text_tokens.to(device)
                text_embeddings = clip_model.encode_text(text_tokens).float()

        pred = model(noised_indices, text_embeddings, r)
        loss = criterion(pred, image_indices)
        loss_adjusted = loss / grad_accum_steps

        loss_adjusted.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5).item()
        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            ema.step_ema(ema_model, model.module)

        acc = (pred.argmax(1) == image_indices).float()
        acc = acc.mean()

        total_loss += loss.item()
        total_acc += acc.item()

        if not proc_id and args.node_id == 0:
            log = {
                "loss": total_loss / (step + 1),
                "acc": total_acc / (step + 1),
                "curr_loss": loss.item(),
                "curr_acc": acc.item(),
                "ppx": np.exp(total_loss / (step + 1)),
                "lr": optimizer.param_groups[0]['lr'],
                "grad_norm": grad_norm
            }
            pbar.set_postfix(log)
            wandb.log(log)

        if args.node_id == 0 and proc_id == 0 and step % args.log_period == 0:
            print(f"Step {step} - loss {total_loss / (step + 1)} - acc {total_acc / (step + 1)} - ppx {np.exp(total_loss / (step + 1))}")

            losses.append(total_loss / (step + 1))
            accuracies.append(total_acc / (step + 1))

            model.eval()
            with torch.no_grad():
                n = 1
                images = images[:10]
                image_indices = image_indices[:10]
                captions = captions[:10]
                text_embeddings = text_embeddings[:10]
                sampled = sample(model.module, c=text_embeddings)[-1]
                sampled_ema = sample(ema_model, c=text_embeddings)[-1]
                sampled = decode(vqmodel, sampled)
                sampled_ema = decode(vqmodel, sampled_ema)
                recon_images = decode(vqmodel, image_indices)

                if args.log_captions:
                    cool_captions_data = torch.load("cool_captions.pth")
                    cool_captions_text = cool_captions_data["captions"]

                    text_tokens = tokenizer.tokenize(cool_captions_text)
                    text_tokens = text_tokens.to(device)
                    cool_captions_embeddings = clip_model.encode_text(text_tokens).float()

                    cool_captions = DataLoader(TensorDataset(cool_captions_embeddings.repeat_interleave(n, dim=0)), batch_size=11)
                    cool_captions_sampled = []
                    cool_captions_sampled_ema = []
                    st = time.time()
                    for caption_embedding in cool_captions:
                        caption_embedding = caption_embedding[0].float().to(device)
                        sampled_text = sample(model.module, c=caption_embedding)[-1]
                        sampled_text_ema = sample(ema_model, c=caption_embedding)[-1]
                        sampled_text = decode(vqmodel, sampled_text)
                        sampled_text_ema = decode(vqmodel, sampled_text_ema)
                        for s, t in zip(sampled_text, sampled_text_ema):
                            cool_captions_sampled.append(s.cpu())
                            cool_captions_sampled_ema.append(t.cpu())
                    print(f"Took {time.time() - st} seconds to sample {len(cool_captions_text) * 2} captions.")

                    cool_captions_sampled = torch.stack(cool_captions_sampled)
                    torchvision.utils.save_image(
                        torchvision.utils.make_grid(cool_captions_sampled, nrow=11),
                        os.path.join(f"results/{args.run_name}", f"cool_captions_{step:03d}.png")
                    )

                    cool_captions_sampled_ema = torch.stack(cool_captions_sampled_ema)
                    torchvision.utils.save_image(
                        torchvision.utils.make_grid(cool_captions_sampled_ema, nrow=11),
                        os.path.join(f"results/{args.run_name}", f"cool_captions_{step:03d}_ema.png")
                    )

                log_images = torch.cat([
                    torch.cat([i for i in sampled.cpu()], dim=-1),
                ], dim=-2)

                log_images_ema = torch.cat([
                    torch.cat([i for i in sampled_ema.cpu()], dim=-1),
                ], dim=-2)

            model.train()

            torchvision.utils.save_image(log_images, os.path.join(f"results/{args.run_name}", f"{step:03d}.png"))
            torchvision.utils.save_image(log_images_ema, os.path.join(f"results/{args.run_name}", f"{step:03d}_ema.png"))

            log_data = [[captions[i]] + [wandb.Image(sampled[i])] + [wandb.Image(sampled_ema[i])] + [wandb.Image(images[i])] + [wandb.Image(recon_images[i])] for i in range(len(captions))]
            log_table = wandb.Table(data=log_data, columns=["Caption", "Image", "EMA", "Orig", "Recon"])
            wandb.log({"Log": log_table})

            if args.log_captions:
                log_data_cool = [[cool_captions_text[i]] + [wandb.Image(cool_captions_sampled[i])] + [wandb.Image(cool_captions_sampled_ema[i])] for i in range(len(cool_captions_text))]
                log_table_cool = wandb.Table(data=log_data_cool, columns=["Caption", "Image", "EMA Image"])
                wandb.log({"Log Cool": log_table_cool})
                del sampled_text, log_data_cool

            del sampled, log_data

            if step % args.extra_ckpt == 0:
                torch.save(model.module.state_dict(), f"models/{args.run_name}/model_{step}.pt")
                torch.save(ema_model.state_dict(), f"models/{args.run_name}/ema_model_{step}.pt")
                torch.save(optimizer.state_dict(), f"models/{args.run_name}/model_{step}_optim.pt")
            torch.save(model.module.state_dict(), f"models/{args.run_name}/model.pt")
            torch.save(ema_model.state_dict(), f"models/{args.run_name}/ema_model.pt")
            torch.save(optimizer.state_dict(), f"models/{args.run_name}/optim.pt")
            torch.save({'step': step, 'losses': losses, 'accuracies': accuracies}, f"results/{args.run_name}/log.pt")

        del images, image_indices, r, text_embeddings
        del noised_indices, mask, pred, loss, loss_adjusted, acc


def launch(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(d) for d in args.devices])
    if len(args.devices) == 1:
        train(0, args)
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "33751"
        p = mp.spawn(train, nprocs=len(args.devices), args=(args,))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "Paella_f8_8192"
    args.model = "UNet"
    args.dataset_type = "webdataset"
    args.total_steps = 501_000
    args.batch_size = 22
    args.image_size = 256
    args.num_workers = 10
    args.log_period = 5000
    args.extra_ckpt = 50_000
    args.accum_grad = 1
    args.num_codebook_vectors = 8192  # 1024
    args.log_captions = True

    args.n_nodes = 8
    args.node_id = int(os.environ["SLURM_PROCID"])
    args.devices = [0, 1, 2, 3, 4, 5, 6, 7]

    args.dataset_path = "pipe:aws s3 cp s3://s-laion/improved-aesthetics-laion-2B-en-subsets/aesthetics_tars/{000000..060207}.tar -"
    print("Launching with args: ", args)
    launch(
        args
    )

import argparse
import datetime
import torch
import wandb
import os 

from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from ddpm import script_utils
from ddpm.celeba_dataset import CelabaDataset, CelebaDataset2, CelebaSketchDataset
# from ddpm.maskVAE import VAE
from common import tensor2map
import numpy as np

from omegaconf import OmegaConf
from ldm_vae.ldm.util import instantiate_from_config

from sketch_ae.models.Combine_AE_dilate import Combine_AE
from sketch_ae.models.Latent_Decoder import Latent_Decoder, One_Latent_Decoder

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def get_model():
    config = OmegaConf.load("ldm_vae/models/first_stage_models/vq-f4/config.yaml")  
    model = load_model_from_config(config, "ldm_vae/models/first_stage_models/vq-f4/model.ckpt")
    return model



def main():
    args = create_argparser().parse_args()
    device = args.device
    # os.environ["WANDB_MODE"]="offline"

    try:

        ae = get_model()

        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)
        Num_Param = sum(p.numel() for p in diffusion.parameters() if p.requires_grad)
        print("Number of Trainable Parameters = %d" % (Num_Param))

        combine_ae = Combine_AE(img_size=512, img_channels=1).to(args.device)
        combine_ae.load_state_dict(torch.load('./sketch_ae/outputs/1028.pt'))
        combine_ae.eval()

        latent_decoder = One_Latent_Decoder(latent_size=128, latent_channels=8).to(args.device)
        optimizer_decoder = torch.optim.Adam(combine_ae.parameters(), lr=args.learning_rate)
        Num_Param_decoder = sum(p.numel() for p in latent_decoder.parameters() if p.requires_grad)
        print("Number of Trainable Parameters = %d" % (Num_Param_decoder))

        if args.model_checkpoint is not None:
            diffusion.load_state_dict(torch.load(args.model_checkpoint))
        if args.optim_checkpoint is not None:
            optimizer.load_state_dict(torch.load(args.optim_checkpoint))
        if args.decoder_checkpoint is not None:
            latent_decoder.load_state_dict(torch.load(args.decoder_checkpoint))

        if args.log_to_wandb:
            if args.project_name is None:
                raise ValueError("args.log_to_wandb set to True but args.project_name is None")

            run = wandb.init(
                project=args.project_name,
                entity='ohicarip',
                config=vars(args),
                name=args.run_name,
            )
            wandb.watch(diffusion)

        batch_size = args.batch_size

         
        transform = transforms.Compose([
            # transforms.RandomSizedCrop(64),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[1, 1, 1])
        ])
        

        # root = './datasets/'
        dataset = CelebaSketchDataset(transform=transform)
        img_inds = np.arange(len(dataset))
        # np.random.shuffle(img_inds)
        train_inds = img_inds[:int(0.9 * len(img_inds))]
        test_inds = img_inds[int(0.9 * len(img_inds)):]

        train_loader = script_utils.cycle(DataLoader(
            dataset,
            batch_size=batch_size,
            # shuffle=True,
            drop_last=True,
            sampler=SubsetRandomSampler(train_inds),
            num_workers=0,
        ))

        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            # shuffle=True,
            drop_last=True,
            sampler=SubsetRandomSampler(test_inds),
            num_workers=0,
        )

        
        acc_train_loss = 0

        for iteration in range(1, args.iterations + 1):
            diffusion.train()
            latent_decoder.train()

            # optimizer.zero_grad()
            # optimizer_decoder.zero_grad()

            train_img, train_sketch = next(train_loader)
                
            train_img = train_img.to(device)
            train_sketch = train_sketch.to(device)

            z, *_= ae.encode(train_img)#.sample()
            # print(z.shape)
            map = latent_decoder(combine_ae.encode(train_sketch))
            loss = diffusion(z, map=map)

            acc_train_loss += loss.item()

            optimizer.zero_grad()
            optimizer_decoder.zero_grad()

            loss.backward()
            optimizer.step()
            optimizer_decoder.step()


            diffusion.update_ema()
            print(iteration)

            if iteration % args.log_rate == 0:
                

                with torch.no_grad():
                    diffusion.eval()
                    latent_decoder.eval()

                    test_loss = 0

                    z = diffusion.sample(args.batch_size, device, map=map)
                    train_samples = ae.decode(z)

                    for test_img, test_sketch in test_loader:
                        test_img = test_img.to(device)
                        test_sketch = test_sketch.to(device)
                        
                        z, *_ = ae.encode(test_img)#.sample()
                        map = latent_decoder(combine_ae.encode(test_sketch))
                        loss = diffusion(z, map=map)

                        test_loss += loss.item()
                
                z = diffusion.sample(args.batch_size, device, map=map)

                test_samples = ae.decode(z)

                # test_samples = test_samples.squeeze().detach().cpu().numpy()
                # test_samples = np.array((test_samples.transpose(0, 2, 3, 1) + 1)/2 *255, np.uint8) # 256
                # test_samples = np.array((test_samples.transpose(1, 2, 0) + 1)/2 *255, np.uint8) # 512
            
                test_loss /= len(test_loader)
                acc_train_loss /= args.log_rate

                wandb.log({
                    "test_loss": test_loss,
                    "train_loss": acc_train_loss,
                    "test_sketch": [wandb.Image(s) for s in test_sketch],
                    "test_samples": [wandb.Image(s) for s in test_samples],
                    "train_sketch": [wandb.Image(input_tmap) for input_tmap in train_sketch],
                    "train_samples": [wandb.Image(tsample) for tsample in train_samples],
                })

                del test_loss, acc_train_loss, test_sketch, test_samples, train_sketch, train_samples
                torch.cuda.empty_cache()

                acc_train_loss = 0
            
            if iteration % args.checkpoint_rate == 0:
                model_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-nocrop-test-iteration-{iteration}-model.pth"
                optim_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-optim.pth"
                decoder_name = f"{args.log_dir}/{args.project_name}-{args.run_name}-nocrop-test-iteration-{iteration}-decoder.pth"

                torch.save(diffusion.state_dict(), model_filename)
                torch.save(latent_decoder.state_dict(), decoder_name)
        
        if args.log_to_wandb:
            run.finish()
    except KeyboardInterrupt:
        if args.log_to_wandb:
            run.finish()
        print("Keyboard interrupt, run finished early")


def create_argparser():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    run_name = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M")
    defaults = dict(
        learning_rate=3e-5, #8
        # learning_rate=4e-6, #1
        batch_size=8,
        iterations=800000,

        log_to_wandb=True,
        log_rate=1000,
        checkpoint_rate=1000,
        log_dir="./ddpm_logs",
        project_name='mask_diff_one_decoder',
        run_name=run_name,

        model_checkpoint=None,
        optim_checkpoint=None,
        decoder_checkpoint=None,

        schedule_low=1e-4, 
        schedule_high=0.02,

        device=device,
    )
    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
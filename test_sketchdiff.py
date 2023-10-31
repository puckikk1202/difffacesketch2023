import argparse
import datetime
import torch
import wandb
import os 
from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from ddpm import script_utils
from ddpm.celeba_dataset import CelabaDataset, CelebaDataset2, CelebaSketchDataset
# from ddpm.maskVAE import VAE
import numpy as np

from omegaconf import OmegaConf
from ldm_vae.ldm.util import instantiate_from_config

from sketch_ae.models.Combine_AE import Combine_AE
from sketch_ae.models.Latent_Decoder import Latent_Decoder

# from torchsummary import summary

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    # print(model)
    model.eval()
    return model

def get_model():
    config = OmegaConf.load("ldm_vae/models/first_stage_models/vq-f4/config.yaml")  
    model = load_model_from_config(config, "ldm_vae/models/first_stage_models/vq-f4/model.ckpt")
    return model



def main():
    args = create_argparser().parse_args()
    device = args.device

    ae = get_model()

    diffusion = script_utils.get_diffusion_from_args(args).to(device)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)
    Num_Param = sum(p.numel() for p in diffusion.parameters() if p.requires_grad)
    print("Number of Trainable Parameters = %d" % (Num_Param))
    print(diffusion)

    combine_ae = Combine_AE(img_size=512, img_channels=1).to(args.device)
    combine_ae.load_state_dict(torch.load(args.ae_ckpt))
    combine_ae.eval()

    latent_decoder = Latent_Decoder(latent_size=128, latent_channels=8).to(args.device)
    optimizer_decoder = torch.optim.Adam(latent_decoder.parameters(), lr=args.learning_rate)
    Num_Param_decoder = sum(p.numel() for p in latent_decoder.parameters() if p.requires_grad)
    print("Number of Trainable Parameters = %d" % (Num_Param_decoder))
    # print(latent_decoder)

    diffusion.load_state_dict(torch.load(args.dm_ckpt))#'./ddpm_logs/0117/sketch_150.pth'
    latent_decoder.load_state_dict(torch.load(args.decoder_ckpt))#'./ddpm_logs/0117/decoder_150.pth'
        

    transform = transforms.Compose([
        # transforms.RandomSizedCrop(64),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5,
                             std=1)
    ])
             
    to_image = transforms.ToPILImage()
    with torch.no_grad():
        diffusion.eval()
        latent_decoder.eval()

        path = args.sketch_path
        for img in tqdm(os.listdir(path)):
            # if int(img.split('.')[0].split('_')[0]) > 8999: 
                for i in range(3):
                    input_sketch = Image.open(path+img).resize((256, 256), Image.LANCZOS).convert('L')
                    input_sketch = transform(input_sketch)
                    # print(input_sketch)
                    input_sketch = input_sketch.unsqueeze(0).to(device)
                    map = latent_decoder(combine_ae.encode(input_sketch)) 
                    z = diffusion.sample(1, device, map=map)

                    output_samples = ae.decode(z)#.squeeze(0)
                    # output_samples = (output_samples+1) / 2
                    # save_img = to_image(output_samples)

                    # wandb.log({"output": [wandb.Image(s) for s in output_samples]})
                    
                    data = output_samples.squeeze(0).detach().cpu().numpy()#.transpose(1, 2, 0)
                    # output_img = np.array((output_img+ 1)/2 *255, np.uint8)
                    # print(np.min(output_img), np.max(output_img))
                    if np.min(data)<0:
                        data = (data-np.min(data)) / np.ptp(data)
                    if np.max(data) <= 1.0:
                        data = (data * 255).astype(np.int32)

                    # data = data.clip(0, 255).astype(np.int32)

                    save_img = Image.fromarray(data.astype(np.uint8).transpose(1, 2, 0))
                    
                    
                    save_img.save(f'./test_draw/0217/output/'+img.split('.')[0]+f'_{str(i)}'+'.jpg')
   

def create_argparser():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    run_name = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M")
    defaults = dict(
        learning_rate=3e-5,
        batch_size=1,
        iterations=800000,

        log_to_wandb=True,
        log_rate=500,
        checkpoint_rate=2000,
        log_dir=None,
        project_name='mask_diff',
        run_name=run_name,

        model_checkpoint=None,
        optim_checkpoint=None,

        schedule_low=1e-4,
        schedule_high=0.02,

        device=device,
        sketch_path='./sketch_example/',
        ae_ckpt=None,
        dm_ckpt=None,
        decoder_ckpt=None,
    )
    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
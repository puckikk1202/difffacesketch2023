import torch
import torch.nn as nn
import os 
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import torchvision
import numpy as np
import argparse
import wandb

from torch.utils.tensorboard import SummaryWriter

from sketch_ae.datasets.celeba_sketch import CelebaSketch
from sketch_ae.models.Combine_AE import Combine_AE
from sketch_ae.models.Combine_AE_dilate import Combine_AE_dilate
from sketch_ae.models.Latent_Decoder import Latent_Decoder

def get_args_parser():

    parser = argparse.ArgumentParser(description='Deep Face Drawing: Train Stage')
    parser.add_argument('--dataset', type=str, default='./datasets/celeba_sketch_nobg/')
    parser.add_argument('--log_dir', type=str, default='./sketch_ae/outputs/')
    parser.add_argument('--log_rate', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=int, default=2e-4)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--img_channels', type=int, default=1)
    parser.add_argument('--kld_w', type=int, default=0.00025)

    parser.add_argument('--checkpoint', type=str, default=None, help='Path to load model weights.')
    parser.add_argument('--output', type=str, default=None, help='Path to save weights.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--date', type=str, default='0608')
    args = parser.parse_args()
    return args

def main():
    model = Combine_AE_dilate(img_size=args.img_size, img_channels=args.img_channels).to(args.device)
    Num_Param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # latent_decoder = Latent_Decoder(latent_size=128, latent_channels=4).to(args.device)
    print("Number of Trainable Parameters = %d" % (Num_Param))

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))
        print(f'model loaded from {args.checkpoint}')

    # run = wandb.init(
    #     project='sketch_vae',
    #     entity='ohicarip',
    #     # config=vars(args),
    #     name='ae_0606',
    # )
    # print('wandb init')
    # wandb.watch(model)
    # print('wandb watch')
    # os.environ["WANDB_MODE"]="online"
    

    writer = SummaryWriter(f'runs/{args.date}')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    loss_fn = nn.MSELoss()

    transform = transforms.Compose([
                # transforms.RandomSizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5],
                                    std=[1])
                ])

    dataset = CelebaSketch(data_dir=args.dataset, transform=transform)
    img_inds = np.arange(len(dataset))

    train_inds = img_inds[:int(0.9 * len(img_inds))]
    test_inds = img_inds[int(0.9 * len(img_inds)):]

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        drop_last=True,
        sampler=SubsetRandomSampler(train_inds),
        num_workers=0,
    )

    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        drop_last=True,
        sampler=SubsetRandomSampler(test_inds),
        num_workers=0,
    )

    encoder_key = ['eye1', 'eye2', 'nose', 'mouth', 'face']
    best_loss = 10

    for epoch in range(args.epochs):
        print(epoch)
        running_loss = {
            'eye1_mse' : 0,
            'eye2_mse' : 0,
            'nose_mse' : 0,
            'mouth_mse' : 0,
            'face_mse' : 0,
            'eye1_kld' : 0,
            'eye2_kld' : 0,
            'nose_kld' : 0,
            'mouth_kld' : 0,
            'face_kld' : 0
        }

        running_log = {
            'eye1' : [],
            'eye2' : [],
            'nose' : [],
            'mouth' : [],
            'face' : [],
            'dilate': []    
        }

        model.train()
        for i, sketches in enumerate(train_loader):
            sketches = sketches.to(args.device)

            for k, key in enumerate(encoder_key):
                if k == 0:
                    target = sketches[:,:, 94:94+64, 54:54+64]
                elif k == 1:
                    target = sketches[:,:, 94:94+64, 128:128+64]
                elif k == 2:
                    target = sketches[:,:, 116:116+96, 91:91+96]
                elif k == 3:
                    target = sketches[:,:, 151:151+96, 85:85+96]
                elif k == 4:
                    target = sketches

            # for k, key in enumerate(encoder_key):
            #     if k == 0:
            #         target = sketches[:,:, 188:188+128, 108:108+128]
            #     elif k == 1:
            #         target = sketches[:,:, 188:188+128, 256:256+128]
            #     elif k == 2:
            #         target = sketches[:,:, 232:232+192, 182:182+192]
            #     elif k == 3:
            #         target = sketches[:,:, 302:302+192, 170:170+192]
            #     elif k == 4:
            #         target = sketches

                output = model(sketches)[k]
                
                # latent = model.encode(sketches)
                # print(latent)
                # c_latent = latent_decoder(latent)
                # print(c_latent.shape)/
                running_log[key] = output[0]
                # print('key', key, running_log[key].shape)
                
                recon_loss = loss_fn(output[0], target)
                kld_loss = output[1]
                loss = recon_loss + args.kld_w*kld_loss
                running_loss[key+'_mse'] = recon_loss
                running_loss[key+'_kld'] = kld_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if k == 4:
                    with torch.no_grad():
                        dilate = model(sketches)[5]
                        # print('dilate', dilate.shape)
                        running_log['dilate'] = dilate
                        # print('dilate',torch.min(dilate), torch.max(dilate))

            

            # if i % log_rate == 0:
        print(f'saving log at: {epoch}.')
        total_loss_mse = running_loss['eye1_mse'] + running_loss['eye2_mse'] + running_loss['nose_mse'] + running_loss['mouth_mse'] + running_loss['face_mse']
        total_loss_kld = running_loss['eye1_kld'] + running_loss['eye2_kld'] + running_loss['nose_kld'] + running_loss['mouth_kld'] + running_loss['face_kld']
        total_loss = total_loss_mse + args.kld_w*total_loss_kld
        print(f'train ------  loss: {total_loss},    recon_loss: {total_loss_mse},      kld_loss: {total_loss_kld}')

   
        sketches_grid = torchvision.utils.make_grid(sketches, normalize=True)
        face_grid = torchvision.utils.make_grid(running_log['face'], normalize=True)
        writer.add_image('train_sketches', sketches_grid, epoch)
        writer.add_image('train_rec', face_grid, epoch)
        writer.add_scalar('train/loss_mse', total_loss_mse, epoch)
        writer.add_scalar('train/loss_kld', args.kld_w*total_loss_kld, epoch)
        
        # wandb.log({
        #     "train_loss_eye1_mse": running_loss['eye1_mse'],
        #     "train_loss_eye2_mse": running_loss['eye2_mse'],
        #     "train_loss_nose_mse": running_loss['nose_mse'],
        #     "train_loss_mouth_mse": running_loss['mouth_mse'],
        #     "train_loss_face_mse": running_loss['face_mse'],
        #     "train_loss_eye1_kld": running_loss['eye1_kld'],
        #     "train_loss_eye2_kld": running_loss['eye2_kld'],
        #     "train_loss_nose_kld": running_loss['nose_kld'],
        #     "train_loss_mouth_kld": running_loss['mouth_kld'],
        #     "train_loss_face_kld": running_loss['face_kld'],
            # "train_sketches": [wandb.Image(sketch) for sketch in sketches],
            # "train_rec": [wandb.Image(rec) for rec in running_log['face']],
            # "train_target_mouth": [wandb.Image(mouth) for mouth in sketches[:,:, 151:151+96, 85:85+96]],
            # # "train_target_mouth": [wandb.Image(mouth) for mouth in sketches[:,:, 302:302+192, 170:170+192]],
            # "train_rec_mouth": [wandb.Image(eye1) for eye1 in running_log['mouth']],
            # "train_rec_dilate": [wandb.Image(eye1) for eye1 in running_log['dilate']]
        # })
        
        model.eval()
        with torch.no_grad():
            testing_loss = {
                'eye1_mse' : 0,
                'eye2_mse' : 0,
                'nose_mse' : 0,
                'mouth_mse' : 0,
                'face_mse' : 0,
                'eye1_kld' : 0,
                'eye2_kld' : 0,
                'nose_kld' : 0,
                'mouth_kld' : 0,
                'face_kld' : 0
                }
            
            testing_log = {
                'eye1' : [],
                'eye2' : [],
                'nose' : [],
                'mouth' : [],
                'face' : [],
                'dilate': []   
            }
            
            if epoch % args.log_rate == 0:
                print(f'saving model at: epoch {epoch}.')
                torch.save(model.state_dict(), f'{args.log_dir}{args.date}/{epoch}.pt')
                
            for i, sketches in enumerate(test_loader):
                if i == 0:
                    sketches = sketches.to(args.device)
                    for k, key in enumerate(encoder_key):
                        if k == 0:
                            target = sketches[:,:, 94:94+64, 54:54+64]
                        elif k == 1:
                            target = sketches[:,:, 94:94+64, 128:128+64]
                        elif k == 2:
                            target = sketches[:,:, 116:116+96, 91:91+96]
                        elif k == 3:
                            target = sketches[:,:, 151:151+96, 85:85+96]
                        elif k == 4:
                            target = sketches

                    # for k, key in enumerate(encoder_key):
                    #     if k == 0:
                    #         target = sketches[:,:, 188:188+128, 108:108+128]
                    #     elif k == 1:
                    #         target = sketches[:,:, 188:188+128, 256:256+128]
                    #     elif k == 2:
                    #         target = sketches[:,:, 232:232+192, 182:182+192]
                    #     elif k == 3:
                    #         target = sketches[:,:, 302:302+192, 170:170+192]
                    #     elif k == 4:
                    #         target = sketches

                        output = model(sketches)[k]
                        testing_log[key] = output[0]
                        recon_loss = loss_fn(output[0], target)
                        kld_loss = output[1]
                        loss = recon_loss + args.kld_w*kld_loss
                        testing_loss[key+'_mse'] = recon_loss
                        testing_loss[key+'_kld'] = kld_loss
                        # testing_loss[key] = loss
                        if k == 4:
                            # with torch.no_grad():
                            dilate = model(sketches)[5]
                            testing_log['dilate'] = dilate
                    
                    
                    # wandb.log({
                    #         "test_loss_eye1_mse": testing_loss['eye1_mse'],
                    #         "test_loss_eye2_mse": testing_loss['eye2_mse'],
                    #         "test_loss_nose_mse": testing_loss['nose_mse'],
                    #         "test_loss_mouth_mse": testing_loss['mouth_mse'],
                    #         "test_loss_face_mse": testing_loss['face_mse'],
                    #         "test_loss_eye1_kld": testing_loss['eye1_kld'],
                    #         "test_loss_eye2_kld": testing_loss['eye2_kld'],
                    #         "test_loss_nose_kld": testing_loss['nose_kld'],
                    #         "test_loss_mouth_kld": testing_loss['mouth_kld'],
                    #         "test_loss_face_kld": testing_loss['face_kld'],
                            # "test_sketches": [wandb.Image(sketch) for sketch in sketches],
                            # "test_rec": [wandb.Image(rec) for rec in testing_log['face']],
                            # "test_target_mouth": [wandb.Image(mouth) for mouth in sketches[:,:, 151:151+96, 85:85+96]],
                            # # "test_target_mouth": [wandb.Image(mouth) for mouth in sketches[:,:, 302:302+192, 170:170+192]],
                            # "test_rec_mouth": [wandb.Image(eye1) for eye1 in testing_log['mouth']],
                            # "test_rec_dilate": [wandb.Image(eye1) for eye1 in testing_log['dilate']]
                        # })

                    total_loss_mse = testing_loss['eye1_mse'] + testing_loss['eye2_mse'] + testing_loss['nose_mse'] + testing_loss['mouth_mse'] + testing_loss['face_mse']
                    total_loss_kld = testing_loss['eye1_kld'] + testing_loss['eye2_kld'] + testing_loss['nose_kld'] + testing_loss['mouth_kld'] + testing_loss['face_kld']
                    total_loss = total_loss_mse + args.kld_w*total_loss_kld
                    print(f'test ------  loss: {total_loss},    recon_loss: {total_loss_mse},      kld_loss: {total_loss_kld}')

                    sketches_grid = torchvision.utils.make_grid(sketches, normalize=True)
                    face_grid = torchvision.utils.make_grid(testing_log['face'], normalize=True)
                    writer.add_image('test_sketches', sketches_grid, epoch)
                    writer.add_image('test_rec', face_grid, epoch)
                    writer.add_scalar('test/loss_mse', total_loss_mse, epoch+1)
                    writer.add_scalar('test/loss_kld', args.kld_w*total_loss_kld, epoch+1)
                    # total_loss = testing_loss['eye1_mse'] + testing_loss['eye2_mse'] + testing_loss['nose_mse'] + testing_loss['mouth_mse'] + testing_loss['face_mse'] + args.kld_w * (testing_loss['eye1_kld'] + testing_loss['eye2_kld'] + testing_loss['nose_kld'] + testing_loss['mouth_kld'] + testing_loss['face_kld'])    
                    if total_loss < best_loss:
                        print(f'saving model at best: epoch {epoch}.')
                        torch.save(model.state_dict(), f'{args.log_dir}{args.date}/best.pt')
                    
                    del running_loss, testing_loss, loss, sketches, total_loss_mse, total_loss_kld, total_loss
    # run.finish()
    writer.close()



if __name__ == "__main__":
    args = get_args_parser()
    main()
'''
-----------------------------------------------
File Name: reggan.py$
Description: 
  5-fold cross validation of semi-supervised deep learning for regression based on Pytroch
  target: Dose prediction on phantom organs
  method: regression gan with/without pseudo label
Author: Jing Zhang
Date: 18/09/2024
-----------------------------------------------
'''
from requirements import *
from data_load import *
from model_load import *
from hyper_parameter import *


if __name__ == "__main__":
    '''
    train reggan 
    '''
    
    message_start = f'baseline model: {baseline}, reggan\n'
    print(message_start)                                           # print to console
    with open(train_log, 'a') as file: file.write(message_start)   # print to log file
    all_r2, all_cc, all_mae, all_mape, all_ps = [], [], [], [], [] # initialization, pseudo labels
    summaryWriter = SummaryWriter('./localruns', flush_secs=1200)

    # 5-fold cross validation
    for folds in range(1,6):
        print(f'================fold {folds}===============================')

        # Initialize models
        generator = Generator().cuda() # a deconv
        discriminator = Discriminator(model_name=baseline, pretrained=pretrain).cuda() # a cnn
        # Optimizers
        optimizer_G = AdamW(generator.parameters(), lr=learn_rate, weight_decay=1e-4)
        optimizer_D = AdamW(discriminator.parameters(), lr=learn_rate, weight_decay=1e-4)
        scheduler_G = ExponentialLR(optimizer_G, gamma=0.99)
        scheduler_D = ExponentialLR(optimizer_D, gamma=0.99)

        # Load data
        train_labeled, train_gt, train_unlabeled, valid_image, valid_gt, test_image, test_gt = data_split_crossval(csv_file, organ, fold = folds, num_label = n_label)
        labeled_set = Dose_Data(data_path=img_path, organ_name=organ, image_list=train_labeled,label_list=train_gt,mode='train', supervised = True)
        unlabel_set = Dose_Data(data_path=img_path, organ_name=organ, image_list=train_unlabeled, mode='train', supervised = False)
        valid_set   = Dose_Data(data_path=img_path, organ_name=organ, image_list=valid_image,label_list=valid_gt,mode='valid')
        test_set    = Dose_Data(data_path=img_path, organ_name=organ, image_list=test_image,label_list=test_gt,mode='test')

        labeled_loader   = DataLoader(dataset=labeled_set, batch_size=batchsize, shuffle=True,  num_workers=6)
        unlabeled_loader = DataLoader(dataset=unlabel_set, batch_size=batchsize, shuffle=True,  num_workers=6)
        valid_loader     = DataLoader(dataset=valid_set,   batch_size=batchsize, shuffle=False, num_workers=6)
        test_loader      = DataLoader(dataset=test_set,    batch_size=batchsize, shuffle=False, num_workers=6)

        print(f'labeled loader {len(labeled_loader)}, unlabeled_loader {len(unlabeled_loader)}, valid_loader {len(valid_loader)}, test_loader {len(test_loader)} ')
        all_num_ps = 0 # number of pseudo labels in all epochs
        best_r2, best_epoch = -1000, 0

        for epoch in range(num_epochs):
            train_losses_d = []
            train_losses_g = []
            num_ps = 0 # inside epoch loop
            discriminator.train()

            for i, (labeled_data, unlabeled_data) in enumerate(zip(labeled_loader, unlabeled_loader)):

                with torch.enable_grad():
                    # Prepare real and fake data for the Discriminator
                    x1_w, y, _ = labeled_data
                    x2_w, wx2_list, x2_s = unlabeled_data
                    x1_w, y = x1_w.cuda(), y.cuda()
                    x2_w, x2_s = x2_w.cuda(), x2_s.cuda()
                    wx2_list = [item.cuda() for item in wx2_list]

                    # Generate fake images
                    z = torch.randn(batchsize, img_height).cuda()  # random noise vector
                    fake_imgs = generator(z)  # fake images
                    true = torch.ones(batchsize, 1).cuda()  # real image label 
                    fake = torch.zeros(batchsize, 1).cuda()  # fake image label

                    # Train Generator to make fake image look very real
                    optimizer_G.zero_grad()
                    fake_pred_g, fake_reg = discriminator(fake_imgs)
                    fake_reg = torch.squeeze(fake_reg, dim=1)
                    g_loss = adversarial_loss(fake_pred_g, true)
                    g_loss.backward(retain_graph=False) # keep gradient, default is false
                    optimizer_G.step()
                    train_losses_g.append(g_loss.item())

                    # Train Discriminator with real labeled data
                    optimizer_D.zero_grad()
                    real1w_pred, real1w_reg = discriminator(x1_w)
                    real1w_reg = torch.squeeze(real1w_reg, dim=1)
                    fake_reg_copy = fake_reg.clone().detach() # only update discriminator
                    real1w_reg_copy = real1w_reg.clone()      # avoid in-place opration

                    # Train Discriminator with real unlabeled data
                    _, real2w_reg = discriminator(x2_w)
                    _, real2s_reg = discriminator(x2_s)
                    real2w_reg = torch.squeeze(real2w_reg, dim=1)
                    real2s_reg = torch.squeeze(real2s_reg, dim=1)

                    d_real_loss = (adversarial_loss(real1w_pred, true) + # supervised loss
                                regression_loss(real1w_reg, y)         # supervised loss
                                + alpha1 * consistency_loss(real1w_reg_copy, fake_reg_copy)  
                                + alpha1 * consistency_loss(real2w_reg, real2s_reg)
                                 )
                    # Train Discriminator with fake data
                    fake_pred_d, _ = discriminator(fake_imgs.detach())  # not update parameters

                    d_fake_loss = adversarial_loss(fake_pred_d, fake) 

                    # Train Discriminator with pseudolabel loss
                    with torch.no_grad():
                        _, p2_s = discriminator(x2_s)
                        weak_stacked = torch.stack(wx2_list, dim=0)
                        list_wx2 = torch.reshape(weak_stacked, (len(wx2_list) * weak_stacked.shape[1], 3, img_height, img_width))
                        _, wp2_list = discriminator(list_wx2)
                        wp2_list_array = torch.chunk(wp2_list, batchsize, dim=0)
                        sp2_array = torch.chunk(p2_s, batchsize, dim=0)
                        l3 = 0
                        n_ps = 0
                        for each in range(len(sp2_array)):
                            wp2_list_temp = wp2_list_array[each]
                            norm_temp = (wp2_list_temp - wp2_list_temp.min()) / (wp2_list_temp.max() - wp2_list_temp.min())
                            if torch.std(norm_temp) < tau:
                                num_ps += 1
                                n_ps += 1
                                pseudo_label = torch.mean(wp2_list_temp)
                                pseudo_label = torch.unsqueeze(pseudo_label, dim=0)
                                l3 += regression_loss(pseudo_label, sp2_array[each])

                    # Ensure n_ps is at least 1 to avoid division by zero
                    n_ps = max(n_ps, 1)
                    total_d_loss = (d_real_loss + d_fake_loss)/2 + alpha2 * l3 / n_ps
                    total_d_loss.backward()
                    optimizer_D.step()
                    train_losses_d.append(total_d_loss.item())
        
            # end of a epoch, print training status
            avg_g_loss = np.array(train_losses_g).mean() # total generator loss for a epoch
            avg_d_loss = np.array(train_losses_d).mean() # total discriminator loss for a epoch

            print(f"[Epoch {epoch+1}/{num_epochs}] [G loss: {avg_g_loss:.4f}] [D loss: {avg_d_loss:.4f}] ")
            summaryWriter.add_scalars('loss', {"train": (avg_d_loss)}, epoch)

            all_num_ps +=num_ps
            torch.cuda.empty_cache()

            # Validation
            discriminator.eval()
            val_labels_list = []
            val_logits_list = []

            with torch.no_grad():
                for x, y in valid_loader:
                    x, y = x.cuda(), y.cuda()
                    _, val_preds = discriminator(x)
                    logits = torch.squeeze(val_preds,dim=1)
                    val_logits_list.extend(logits.detach().cpu())
                    val_labels_list.extend(y)

            r2 = r2score(torch.tensor(val_logits_list), torch.tensor(val_labels_list)).item()
            cc = ccscore(torch.tensor(val_logits_list), torch.tensor(val_labels_list)).item()
            avg_mae = mae(torch.tensor(val_logits_list), torch.tensor(val_labels_list)).item()
            avg_mape = mape(torch.tensor(val_logits_list), torch.tensor(val_labels_list)).item()

            print("[VALID] epoch={}/{}  val_r2={:.3f} val_cc={:.3f} val_mae={:.3f} val_mape={:.3f}".format(epoch+1, num_epochs, r2, cc, avg_mae, avg_mape))   
            summaryWriter.add_scalars('r2', {"val": r2}, epoch)
            summaryWriter.add_scalars('mae', {"val": avg_mae}, epoch)
            scheduler_G.step() # adjust learning rate
            scheduler_D.step()
            if r2 >= best_r2:
                print(f'best r2 {r2:.3f} at epoch {epoch+1}')
                best_r2 = r2
                best_epoch = epoch
                torch.save(discriminator.state_dict(), best_model_path+str(folds)+baseline+'_model.pth')

        print(f"Training finished. The best model saved at epoch {best_epoch}\n")
        ps_epoch = all_num_ps / num_epochs
        all_ps.append(ps_epoch)

        # End of training

        print('################# TRAIN FINISH, START TEST #################')
        discriminator.load_state_dict(torch.load(best_model_path+str(folds)+baseline+'_model.pth'))
        discriminator.eval()
        test_loader = DataLoader(dataset=test_set, batch_size=batchsize, shuffle=False, num_workers=6, pin_memory=True)
        test_logits_list = []
        test_labels_list = []

        with torch.no_grad():
            for x, y, _ in test_loader:
                x, y = x.cuda(), y.cuda()
                _, test_preds = discriminator(x)
                logits = torch.squeeze(test_preds,dim=1)
                test_logits_list.extend(logits.detach().cpu())
                test_labels_list.extend(y)
                    
        r2 = r2score(torch.tensor(test_logits_list), torch.tensor(test_labels_list)).item()
        cc = ccscore(torch.tensor(test_logits_list), torch.tensor(test_labels_list)).item()
        mean_mae = mae(torch.tensor(test_logits_list), torch.tensor(test_labels_list)).item()
        mean_mape = mape(torch.tensor(test_logits_list), torch.tensor(test_labels_list)).item()

        message_test = f'[TEST] fold:{folds} test_r2={r2:.3f} test_cc={cc:.3f} test_mae={mean_mae:.3f} test_mape={mean_mape:.3f}\n'
        print(message_test) # print to console
        with open(train_log, 'a') as file: file.write(message_test) # print to log file

        all_r2.append(r2)
        all_cc.append(cc)
        all_mae.append(mean_mae)
        all_mape.append(mean_mape)

    avg_r2 = statistics.mean(all_r2)
    std_r2 = statistics.stdev(all_r2)
    avg_cc = statistics.mean(all_cc)
    std_cc = statistics.stdev(all_cc)
    avg_mae = statistics.mean(all_mae)
    std_mae = statistics.stdev(all_mae)
    avg_mape = statistics.mean(all_mape)
    std_mape = statistics.stdev(all_mape)


    print('\n\n**************5-Fold results*************************************************\n')
    message_all_folds = f'average number of pseudo labels {int(statistics.mean(all_ps))}\n \
r2 {avg_r2:.3f} \t {std_r2:.3f}\t cc {avg_cc:.3f} \t {std_cc:.3f}\t mape {avg_mape:.3f} \t {std_mape:.3f} mae {avg_mae:.3f} \t {std_mae:.3f}\n\
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\
*************************************************************************************\n\n'
    print(message_all_folds) # print to console
    with open(train_log, 'a') as file: file.write(message_all_folds)  # print to log file
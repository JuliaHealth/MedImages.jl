'''
-----------------------------------------------
File Name: regfixmatch.py$
Description: 
  5-fold cross validation of semi-supervised deep learning for regression based on Pytroch
  target: Dose prediction on phantom organs
  method: regression fixmatch (https://github.com/YongwonJo/SimRegMatch/blob/main/tasks/SimRegMatchTrainer.py)
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
    train SimRegMatch 
    '''
    
    message_start = f'baseline model: {baseline}, SimRegMatch\n'
    print(message_start)                                           # print to console
    with open(train_log, 'a') as file: file.write(message_start)   # print to log file
    all_r2, all_cc, all_mae, all_mape, all_ps = [], [], [], [], [] # initialization, pseudo labels
    summaryWriter = SummaryWriter('./localruns', flush_secs=1200)

    # 5-fold cross validation
    for folds in range(1,6):
        print(f'================fold {folds}===============================')
        
        model = resnet50(dropout=0.1)
        model.cuda()
        optimizer = opt(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = ExponentialLR(optimizer, gamma=0.99)
        train_labeled, train_gt, train_unlabeled, valid_image, valid_gt, test_image, test_gt = data_split_crossval(csv_file, organ, fold = folds, num_label = n_label)
        labeled_set   = Dose_Data(data_path=img_path, organ_name=organ, image_list=train_labeled, label_list=train_gt, mode='train', supervised = True)
        unlabel_set   = Dose_Data(data_path=img_path, organ_name=organ, image_list=train_unlabeled,                    mode='train', supervised = False)
        valid_set     = Dose_Data(data_path=img_path, organ_name=organ, image_list=valid_image,   label_list=valid_gt, mode='valid')
        test_set      = Dose_Data(data_path=img_path, organ_name=organ, image_list=test_image,    label_list=test_gt,  mode='test')

        labeled_loader   = DataLoader(dataset=labeled_set, batch_size=batchsize, shuffle=True,  num_workers=6)
        unlabeled_loader = DataLoader(dataset=unlabel_set, batch_size=batchsize, shuffle=True,  num_workers=6)
        valid_loader     = DataLoader(dataset=valid_set,   batch_size=batchsize, shuffle=False, num_workers=6)
        test_loader      = DataLoader(dataset=test_set,    batch_size=batchsize, shuffle=False, num_workers=6)

        all_num_ps = 0 # number of pseudo labels in all epochs
        best_r2, best_epoch = -1000, 0

        for epoch in range(num_epochs):
            train_losses = []
            num_ps = 0 # inside epoch loop
            model.train()

            for labeled_data, unlabeled_data in zip_longest(labeled_loader, unlabeled_loader, fillvalue=None):

                with torch.enable_grad():
                    model.zero_grad()
                    
                    l1 = torch.tensor(0.0, requires_grad=True).cuda()
                    l3 = torch.tensor(0.0, requires_grad=True).cuda()
                    
                    if labeled_data is not None:
                        x1, gt = labeled_data
                        x1, gt = x1.cuda(), gt.cuda()
                        p1, vecs_x = model(x1)
                        p1 = torch.squeeze(p1, dim=1)
                        l1 = criterion(gt, p1)
                    
                    if unlabeled_data is not None:
                        wx2, wx2_list, sx2 = unlabeled_data
                        wx2, sx2 = wx2.cuda(), sx2.cuda()
                        for item in range(len(wx2_list)): wx2_list[item] = wx2_list[item].cuda()

                        # Predict weak-augmented examples 
                        weak_stacked = torch.stack(wx2_list, dim=0)  # [10, bs, 3, 224, 224]
                        list_wx2 = torch.reshape(weak_stacked, (len(wx2_list) * weak_stacked.shape[1], 3, img_height, img_width))
                        wp2_list, wv2_list = model(list_wx2)
                        wp2_list_array = torch.chunk(wp2_list, batchsize, dim=0)  # torch.Size([10, bs, 1])
                        wv2_list_array = torch.chunk(wv2_list, batchsize, dim=0)  # torch.Size([10, bs, 2048])

                        vecs_w = torch.stack(wv2_list_array, dim=0)  # convert list to tensor
                        vecs_w = torch.mean(vecs_w, dim=1)  

                        # Predict strong-augmented examples
                        preds_s, _ = model(sx2)
                        preds_s = torch.squeeze(preds_s, dim=1)

                        sp2_array = torch.chunk(preds_s, batchsize, dim=0)
                        # similarity distribution
                        vecs_w, vecs_x = F.normalize(vecs_w, dim=1), F.normalize(vecs_x, dim=1)
                        similaritys = vecs_w @ vecs_x.T
                        similaritys = torch.softmax(similaritys / 1, dim=1)

                        # similarity-based pseudo-label
                        v_similarity = similaritys @ gt
                        n_ps = 0 # inside batch loop
                        for each in range(len(sp2_array)):  # each image(10 predictions) in a batch(10)
                            wp2_list_temp = wp2_list_array[each]
                            norm_temp = (wp2_list_temp - wp2_list_temp.min()) / (wp2_list_temp.max() - wp2_list_temp.min())

                            pseudo_label = torch.mean(wp2_list_array[each])  # initial psl
                            # pseudo-label calibration
                            pseudo_label = 0.6 * pseudo_label + (1 - 0.6) * v_similarity[each]  # beta*modelPL + (1-beta)*simPL

                            if torch.std(norm_temp) < tau:
                                num_ps += 1
                                n_ps +=1
                                sp2 = torch.squeeze(sp2_array[each], dim=0)
                                l3 += criterion(pseudo_label.detach(), sp2) # no grad back to model
                    # Ensure n_ps is at least 1 to avoid division by zero
                    n_ps = max(n_ps, 1)
                    total_loss = l1 + alpha2 * l3 / n_ps  # total loss for a batch, alpha2=0.01 in original paper
                    total_loss.backward()
                    optimizer.step()
                    train_losses.append(total_loss.item())

            avg_loss = np.array(train_losses).mean() # total loss for a epoch
            print("[TRAIN] epoch={}/{} train_loss={:.3f}".format(epoch+1, num_epochs, avg_loss))
            summaryWriter.add_scalars('loss', {"train": (avg_loss)}, epoch)
            all_num_ps +=num_ps
            # valid
            torch.cuda.empty_cache()
            model.eval()
            val_labels_list = []
            val_logits_list = []

            with torch.no_grad():
                for x, y in valid_loader:

                    x = x.cuda()
                    logits,_ = model(x)
                    logits = torch.squeeze(logits,dim=1)
                    val_logits_list.extend(logits.detach().cpu())
                    val_labels_list.extend(y)
                
            r2 = r2score(torch.tensor(val_logits_list),torch.tensor(val_labels_list)).item()
            cc = ccscore(torch.tensor(val_logits_list),torch.tensor(val_labels_list)).item()
            avg_mae  = mae(torch.tensor(val_logits_list),torch.tensor(val_labels_list)).item()
            avg_mape = mape(torch.tensor(val_logits_list),torch.tensor(val_labels_list)).item()
            print("[VALID] epoch={}/{}  val_r2={:.3f} val_cc={:.3f} val_mae={:.3f} val_mape={:.3f}".format(epoch+1, num_epochs, r2, cc, avg_mae, avg_mape))   
            summaryWriter.add_scalars('r2', {"val": r2}, epoch)
            summaryWriter.add_scalars('mae', {"val": avg_mae}, epoch)
            scheduler.step()
            if r2 >= best_r2:
                print(f'best r2 {r2:.3f} at epoch {epoch+1}')
                best_r2 = r2
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path+str(folds)+'_SimRegMatch.pth')
        print(f"Training finished. The best model saved at epoch {best_epoch+1}\n")
        # end of training
        ps_epoch = all_num_ps/num_epochs
        all_ps.append(ps_epoch)

        print('################# TRAIN FINISH, START TEST #################')
        model.load_state_dict(torch.load(best_model_path+str(folds)+'_SimRegMatch.pth'))
        model.eval()
        test_loader = DataLoader(dataset=test_set,batch_size=batchsize,shuffle=False,num_workers=6,pin_memory=True)
        test_logits_list = []
        test_labels_list = []
        with torch.no_grad():
            for x, y, _ in test_loader:
                x = x.cuda()
                logits,_ = model(x)
                logits = torch.squeeze(logits,dim=1)
                test_logits_list.extend(logits.detach().cpu())
                test_labels_list.extend(y)
                    
        r2 = r2score(torch.tensor(test_logits_list),torch.tensor(test_labels_list)).item()
        cc = ccscore(torch.tensor(test_logits_list),torch.tensor(test_labels_list)).item()
        mean_mae = mae(torch.tensor(test_logits_list),torch.tensor(test_labels_list)).item()
        mean_mape = mape(torch.tensor(test_logits_list),torch.tensor(test_labels_list)).item()
        
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
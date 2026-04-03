from requirements import *

def data_split_crossval(file, organ, fold = 1, num_label = 40, aff_info = True):
    ''' 
    5-fold cross validation (total 1000 images)
    organ: bladder liver pancreas kidneys spleen prostate rectum salivary
    fold0=fold1=fold2=fold3=fold4=fold5=fold6=fold7=fold8=fold9=100
    train: labeled:40,200,400,600,790
    valid: 100
    test:  100
    return: 
        train_img_labeled, train_groundtruth
        train_img_unlabel,
        val_image, val_label
        test_image, test_label
    '''

    dataframe = pd.read_csv(file)

    fold0 = dataframe[dataframe['fold']==0]
    fold1 = dataframe[dataframe['fold']==1]
    fold2 = dataframe[dataframe['fold']==2]
    fold3 = dataframe[dataframe['fold']==3]
    fold4 = dataframe[dataframe['fold']==4]
    fold5 = dataframe[dataframe['fold']==5] 
    fold6 = dataframe[dataframe['fold']==6]
    fold7 = dataframe[dataframe['fold']==7]
    fold8 = dataframe[dataframe['fold']==8]
    fold9 = dataframe[dataframe['fold']==9]

    fold0_image = fold0['phantoms'].tolist()
    fold1_image = fold1['phantoms'].tolist()
    fold2_image = fold2['phantoms'].tolist()
    fold3_image = fold3['phantoms'].tolist()
    fold4_image = fold4['phantoms'].tolist()
    fold5_image = fold5['phantoms'].tolist()
    fold6_image = fold6['phantoms'].tolist()
    fold7_image = fold7['phantoms'].tolist()
    fold8_image = fold8['phantoms'].tolist()
    fold9_image = fold9['phantoms'].tolist()

    fold0_label = fold0['dose_'+organ].tolist()
    fold1_label = fold1['dose_'+organ].tolist()
    fold2_label = fold2['dose_'+organ].tolist()
    fold3_label = fold3['dose_'+organ].tolist()
    fold4_label = fold4['dose_'+organ].tolist()
    fold5_label = fold5['dose_'+organ].tolist()
    fold6_label = fold6['dose_'+organ].tolist()
    fold7_label = fold7['dose_'+organ].tolist()
    fold8_label = fold8['dose_'+organ].tolist()
    fold9_label = fold9['dose_'+organ].tolist()

    if fold == 1:
        train_image = fold2_image + fold3_image + fold4_image + fold5_image + fold6_image + fold7_image + fold8_image + fold9_image
        train_label = fold2_label + fold3_label + fold4_label + fold5_label + fold6_label + fold7_label + fold8_label + fold9_label
        
        train_img_labeled = train_image[:num_label]
        train_img_unlabel = train_image[num_label:]
        train_groundtruth = train_label[:num_label]

        val_image  = fold0_image
        val_label  = fold0_label
        test_image = fold1_image
        test_label = fold1_label

    if fold == 2:
        train_image = fold0_image + fold3_image + fold4_image + fold5_image + fold6_image + fold7_image + fold8_image + fold9_image
        train_label = fold0_label + fold3_label + fold4_label + fold5_label + fold6_label + fold7_label + fold8_label + fold9_label
        
        train_img_labeled = train_image[:num_label]
        train_img_unlabel = train_image[num_label:]
        train_groundtruth = train_label[:num_label]

        val_image  = fold1_image
        val_label  = fold1_label
        test_image = fold2_image
        test_label = fold2_label

    if fold == 3:
        train_image = fold0_image + fold1_image + fold4_image + fold5_image + fold6_image + fold7_image + fold8_image + fold9_image
        train_label = fold0_label + fold1_label + fold4_label + fold5_label + fold6_label + fold7_label + fold8_label + fold9_label
        
        train_img_labeled = train_image[:num_label]
        train_img_unlabel = train_image[num_label:]
        train_groundtruth = train_label[:num_label]

        val_image  = fold2_image
        val_label  = fold2_label
        test_image = fold3_image
        test_label = fold3_label

    if fold == 4:
        train_image = fold0_image + fold1_image + fold2_image + fold5_image + fold6_image + fold7_image + fold8_image + fold9_image
        train_label = fold0_label + fold1_label + fold2_label + fold5_label + fold6_label + fold7_label + fold8_label + fold9_label
        
        train_img_labeled = train_image[:num_label]
        train_img_unlabel = train_image[num_label:]
        train_groundtruth = train_label[:num_label]

        val_image  = fold3_image
        val_label  = fold3_label
        test_image = fold4_image
        test_label = fold4_label

    if fold == 5:
        train_image = fold0_image + fold1_image + fold2_image + fold3_image + fold6_image + fold7_image + fold8_image + fold9_image
        train_label = fold0_label + fold1_label + fold2_label + fold3_label + fold6_label + fold7_label + fold8_label + fold9_label
        
        train_img_labeled = train_image[:num_label]
        train_img_unlabel = train_image[num_label:]
        train_groundtruth = train_label[:num_label]

        val_image  = fold4_image
        val_label  = fold4_label
        test_image = fold5_image
        test_label = fold5_label
        
    if aff_info == True:
        print("Total training images: {},  labeled: {}, unlabeled: {},  val: {}, test: {}".\
        format(len(train_img_labeled + train_img_unlabel), len(train_img_labeled), len(train_img_unlabel), len(val_image), len(test_image)))

    return train_img_labeled, train_groundtruth, train_img_unlabel, val_image, val_label, test_image, test_label


class Dose_Data(Dataset):
    
    def __init__(self, data_path, organ_name, image_list='', label_list='', mode='train', supervised=True):
        self.data_path = data_path
        self.organ_name = organ_name
        self.image_list = image_list
        self.label_list = label_list
        self.mode = mode
        self.supervised = supervised

        # Augmentation types
        channel_stats = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(**channel_stats)
        
        self.hflip = transforms.RandomHorizontalFlip()
        self.rotat = transforms.RandomRotation(5)
        self.clr_j = transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5)
        self.noise = transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5))
        self.persp = transforms.RandomPerspective(distortion_scale=0.5)
        
        # Weak and strong augmentations
        self.weak_aug = transforms.Compose([self.hflip, self.rotat, self.totensor, self.normalize])
        self.strong_aug = transforms.Compose([self.hflip, self.rotat, self.clr_j, self.noise, self.persp, self.totensor, self.normalize])
        self.transform_norm = transforms.Compose([self.totensor, self.normalize])  # for valid and test

        # Generate a list of weak augmentations with random transformations
        self.transform_weak_list = []
        for _ in range(10):
            transform_w = transforms.Compose([self.hflip, self.rotat, self.totensor, self.normalize])
            transform_w.transforms[0].p = 0.5 + random.uniform(-0.1, 0.1)  # random probability
            self.transform_weak_list.append(transform_w)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]

        # Load CT, mask, and PET images
        ct = Image.open(self.data_path + 'ct/' + image_name + '.png')
        mask = Image.open(self.data_path + 'mask/' + self.organ_name + '/' + image_name + '.png')
        pet = sitk.ReadImage(self.data_path + 'pet/' + image_name + '.mhd')
        pet = sitk.GetArrayFromImage(pet)
        pet = ((pet - np.min(pet)) / (np.max(pet) - np.min(pet)) * 255).astype(np.uint8)

        # Stack into 3 channels and normalize
        array = np.stack((ct, pet, mask), axis=-1)
        array = ((array - np.min(array)) / (np.max(array) - np.min(array)) * 255).astype(np.uint8)
        image = Image.fromarray(array)

        # Apply augmentations based on mode
        if self.mode == 'train':
            if self.supervised:
                # Apply weak augmentation
                x1 = self.weak_aug(image)
                label = self.label_list[idx]
                return x1, np.float32(label)
            else:
                # Apply weak and strong augmentations for unsupervised training
                x2_w = self.weak_aug(image)
                x2_list = [aug(image) for aug in self.transform_weak_list]
                x2_s = self.strong_aug(image)
                return x2_w, x2_list, x2_s
        
        if self.mode == 'valid':
            # Apply normalization for validation
            image = self.transform_norm(image)
            label = self.label_list[idx]
            return image, np.float32(label)

        if self.mode == 'test':
            # Apply normalization for testing and return image_name
            image = self.transform_norm(image)
            label = self.label_list[idx]
            return image, np.float32(label), image_name
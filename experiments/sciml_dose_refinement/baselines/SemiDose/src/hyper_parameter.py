from requirements import *

torch.manual_seed(42)   # Set fixed random number seed
best_model_path = './'
img_path        = "path/wholebody_2D/"
csv_file        = img_path + 'dose_organs.csv'
train_log       = 'train_log.txt'

baseline   = 'resnet50' # 'caformer_s36'
pretrain   = False if baseline == 'resnet50' else True
num_epochs = 2
MeanTEpoch = 100
learn_rate = 1e-4
batchsize  = 10    # batch size

#organ = 'bladder'
#organ = 'kidneys'
#organ = 'liver'
#organ = 'spleen'
#organ = 'pancreas'
#organ = 'prostate'
#organ = 'rectum'
organ = 'salivary'

n_label    = 40        # 40, 200, 400, 600, 790
alpha1     = 0.1       # consistency  loss
alpha2     = 0.1       # pseudolabel loss
beta       = 0.1       # featurematching loss in reggan
tau        = 0.25      # pseudo label threshold
supervised = False     # if supervised training
if supervised==True: alpha1=alpha2=0 
img_height = img_width = 256

criterion  = nn.L1Loss()
opt        = AdamW
mae        = MeanAbsoluteError()
mape       = MeanAbsolutePercentageError()
r2score    = R2Score()
ccscore    = PearsonCorrCoef()

# Adversarial loss in reggan
consistency_loss = nn.L1Loss()
adversarial_loss = nn.BCELoss() #nn.BCELoss() # nn.MSELoss()
feature_matching_loss = nn.MSELoss()
regression_loss = nn.L1Loss()
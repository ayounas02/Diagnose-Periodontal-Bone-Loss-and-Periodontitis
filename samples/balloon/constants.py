

perio_release_file = './dataset/Labelled json files_Perio1-v0.1.json'
cej_release_file = './dataset/Labelled json files_CEJ-v0.1.json'
teeth_release_file = './dataset/Labelled json files_labelled xrays_labelled.json'


learning_rate = 0.0001
train_batch_size = 10

train_directory = 'Braindata/TRAINING'
#train_directory = 'Braindata/TRAINING_old'
test_directory = 'Braindata/TEST'

multiple_inputs = False  ## if set to True, just take one image as an input else take 25
single_in_channels = 1
multiple_in_channels = 25
out_channels = 1
LOAD_MODEL = False
#BATCH_SIZE = 15
BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
WEIGHT_DECAY = 0
NUM_EPOCHS = 3
NUM_WORKERS = 0
PIN_MEMORY = False

PERIO_RESULT_DIR = "../teeth/result/perio/"
CEJ_RESULT_DIR = "../teeth/result/cej/"

##### mask & flair resizing
width = 160
height = 240
# width = 320
# height = 260

##### checkpoints path for resuming training
PERIO_LOAD = False
CEJ_LOAD = False

PERIO_SAVED_DIR = './checkpoints/perio/'
CEJ_SAVED_DIR = './checkpoints/cej/'

PERIOD_MODEL_LOAD_PATH = 'checkpoints/perio/7.pth.tar'
CEJ_MODEL_LOAD_PATH = 'checkpoints/cej/7.pth.tar'








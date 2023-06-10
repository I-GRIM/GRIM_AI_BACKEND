#!/usr/bin/env python
# coding: utf-8

# # Test

# # Dependencies

#     %cd /home/super/Desktop/dreambooth/content
#     !pip install -qq --no-deps accelerate==0.12.0
#     !wget -q -i https://raw.githubusercontent.com/TheLastBen/fast-stable-diffusion/main/Dependencies/dbdeps.txt
#     !sudo dpkg -i *.deb
#     !sudo tar -C / --zstd -xf gcolabdeps.tar.zst
#     !rm *.deb | rm *.zst | rm *.txt
#     !git clone -q --depth 1 --branch main https://github.com/TheLastBen/diffusers
#     !pip install gradio==3.16.2 --no-deps -qq  
#     !pip install -U xformers --no-deps -qq
#     %env LD_PRELOAD=libtcmalloc.so
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # export TF_CPP_MIN_LOG_LEVE='3'
#     os.environ['PYTHONWARNINGS'] = 'ignore'  
# 
#     # additionally add MyDrive folder in content
#     # git clone https://github.com/TheLastBen/fast-stable-diffusion.git
# 

# # Download Model
# set model to defaul diffusion
# 
# in project content directory
# $ sudo mkdir stable-diffusion-v1-5
# $ cd stable-diffusion-v1-5
# $ sudo apt-get install git-lfs
# may have to set git ssh
# 
# !git init
# !git lfs install --system --skip-repo
# !git remote add -f origin  "https://huggingface.co/runwayml/stable-diffusion-v1-5"
# !git config core.sparsecheckout true
# 
# !echo -e "scheduler\ntext_encoder\ntokenizer\nunet\nvae\nmodel_index.json\n!vae/diffusion_pytorch_model.bin\n!*.safetensors" > .git/info/sparse-checkout
# # I get .git/info/sparse-checkout permission denied erro here
# don't use echo, use vim to create sparse-checkout then pull
# 
# !git pull origin main
# !wget -q -O vae/diffusion_pytorch_model.bin https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin
# !rm -r .git
# !rm model_index.json
# !wget 'https://raw.githubusercontent.com/TheLastBen/fast-stable-diffusion/main/Dreambooth/model_index.json'
# %cd /content

# In[2]:


import os
from IPython.display import clear_output
from IPython.utils import capture
from os import listdir
from os.path import isfile
from subprocess import check_output
import time

MODEL_NAME="/home/super/Desktop/dreambooth/content/stable-diffusion-v1-5"
  
PT=""

Session_Name = "character" 

WORKSPACE='/home/super/Desktop/dreambooth/content/MyDrive/Fast-Dreambooth'

# if Session_Link_optional !="":
#   print('[1;32mDownloading session...')
#   with capture.capture_output() as cap:
#     %cd /content
#     if not os.path.exists(str(WORKSPACE+'/Sessions')):
#       %mkdir -p $WORKSPACE'/Sessions'
#       time.sleep(1)
#     %cd $WORKSPACE'/Sessions'
#     !gdown --folder --remaining-ok -O $Session_Name  $Session_Link_optional
#     %cd $Session_Name
#     !rm -r instance_images
#     !unzip instance_images.zip
#     !rm -r concept_images
#     !unzip concept_images.zip
#     !rm -r captions
#     !unzip captions.zip
#     %cd /content


INSTANCE_NAME=Session_Name
OUTPUT_DIR="/home/super/Desktop/dreambooth/content/models/"+Session_Name
SESSION_DIR=WORKSPACE+'/Sessions/'+Session_Name
INSTANCE_DIR=SESSION_DIR+'/instance_images'
CONCEPT_DIR=SESSION_DIR+'/concept_images'
CAPTIONS_DIR=SESSION_DIR+'/captions'
MDLPTH=str(SESSION_DIR+"/"+Session_Name+'.ckpt')

if os.path.exists(str(SESSION_DIR)):
  mdls=[ckpt for ckpt in listdir(SESSION_DIR) if ckpt.split(".")[-1]=="ckpt"]
  if not os.path.exists(MDLPTH) and '.ckpt' in str(mdls):  
    
    def f(n):
      k=0
      for i in mdls:
        if k==n:
          get_ipython().system('mv "$SESSION_DIR/$i" $MDLPTH')
        k=k+1

    k=0
    print('[1;33mNo final checkpoint model found, select which intermediary checkpoint to use, enter only the number, (000 to skip):\n[1;34m')

    for i in mdls:
      print(str(k)+'- '+i)
      k=k+1
    n=input()
    while int(n)>k-1:
      n=input()
    if n!="000":
      f(int(n))
      print('[1;32mUsing the model '+ mdls[int(n)]+" ...")
      time.sleep(2)
    else:
      print('[1;32mSkipping the intermediary checkpoints.')
    del n

with capture.capture_output() as cap:
  get_ipython().run_line_magic('cd', '/home/super/Desktop/dreambooth/content')
  resume=False

if os.path.exists(str(SESSION_DIR)) and not os.path.exists(MDLPTH):
  print('[1;32mLoading session with no previous model, using the original model or the custom downloaded model')
  if MODEL_NAME=="":
    print('[1;31mNo model found, use the "Model Download" cell to download a model.')
  else:
    print('[1;32mSession Loaded, proceed to uploading instance images')

elif os.path.exists(MDLPTH):
  print('[1;32mSession found, loading the trained model ...')
  print('[1;33mDetecting model version...')
  Model_Version=check_output('python3 det.py --MODEL_PATH '+MDLPTH, shell=True).decode('utf-8').replace('\n', '')
  clear_output()
  print('[1;32m'+Model_Version+' Detected') 
  get_ipython().system('rm det.py  ')
  if Model_Version=='1.5':
    get_ipython().system('wget -q -O config.yaml https://github.com/CompVis/stable-diffusion/raw/main/configs/stable-diffusion/v1-inference.yaml')
    print('[1;32mSession found, loading the trained model ...')
    get_ipython().system('python3 /home/super/Desktop/dreambooth/content/diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path $MDLPTH --dump_path "$OUTPUT_DIR" --original_config_file config.yaml')
    get_ipython().system('rm /home/super/Desktop/dreambooth/content/config.yaml')

  elif Model_Version=='V2.1-512px':
    get_ipython().system('wget -q -O convertodiff.py https://raw.githubusercontent.com/TheLastBen/fast-stable-diffusion/main/Dreambooth/convertodiffv2.py')
    print('[1;32mSession found, loading the trained model ...')
    get_ipython().system('python3 /home/super/Desktop/dreambooth/content/convertodiff.py "$MDLPTH" "$OUTPUT_DIR" --v2 --reference_model stabilityai/stable-diffusion-2-1-base')
    get_ipython().system('rm /home/super/Desktop/dreambooth/content/convertodiff.py')

  elif Model_Version=='V2.1-768px':
    get_ipython().system('wget -q -O convertodiff.py https://github.com/TheLastBen/fast-stable-diffusion/raw/main/Dreambooth/convertodiffv2-768.py')
    print('[1;32mSession found, loading the trained model ...')
    get_ipython().system('python3 /home/super/Desktop/dreambooth/content/convertodiff.py "$MDLPTH" "$OUTPUT_DIR" --v2 --reference_model stabilityai/stable-diffusion-2-1')
    get_ipython().system('rm /home/super/Desktop/dreambooth/content/convertodiff.py')
  
  
  if os.path.exists(OUTPUT_DIR+'/unet/diffusion_pytorch_model.bin'):
    resume=True
    clear_output()
    print('[1;32mSession loaded.')
  else:     
    if not os.path.exists(OUTPUT_DIR+'/unet/diffusion_pytorch_model.bin'):
      print('[1;31mConversion error, if the error persists, remove the CKPT file from the current session folder')

elif not os.path.exists(str(SESSION_DIR)):
    get_ipython().run_line_magic('mkdir', '-p "$INSTANCE_DIR"')
    print('[1;32mCreating session...')
    if MODEL_NAME=="":
      print('[1;31mNo model found, use the "Model Download" cell to download a model.')
    else:
      print('[1;32mSession created, proceed to uploading instance images')

    #@markdown

    #@markdown # The most important step is to rename the instance pictures of each subject to a unique unknown identifier, example :
    #@markdown - If you have 10 pictures of yourself, simply select them all and rename only one to the chosen identifier for example : phtmejhn, the files would be : phtmejhn (1).jpg, phtmejhn (2).png ....etc then upload them, do the same for other people or objects with a different identifier, and that's it.
    #@markdown - Checkout this example : https://i.imgur.com/d2lD3rz.jpeg


# In[3]:


# Image convert to bytes image
import PIL
from io import BytesIO

def PIL_convert(img_path , quality =100) :
    img = Image.open(img_path).convert('RGB')
    buffer = BytesIO()
    img.save(buffer, format='png',quality = quality)

    val = buffer.getvalue()
    PIL_img = PIL.Image.open(buffer)
    return val, PIL_img



from io import BytesIO
from PIL import Image

def image_refer() :
    imge_dir = '/home/super/Desktop/dreambooth/test'
    img_path_dict  = {}
    img_path_list = []
    possible_img_extension = ['.jpg', '.jpeg', '.JPG', '.bmp', '.png'] # 이미지 확장자들
    
    for (root, dirs, files) in os.walk(imge_dir):
        if len(files) > 0:
            for file_name in files:
                if os.path.splitext(file_name)[1] in possible_img_extension:
                    img_path = root + '/' + file_name
                    # 경로에서 \를 모두 /로 바꿔줘야함
                    img_path = img_path.replace('\\', '/') # \는 \\로 나타내야함         
                    img_path_list.append(img_path)
                    img_byte, img = PIL_convert(img_path)
                    img_path_dict[img_path] = img_byte
    return img_path_dict
img_dict_list = image_refer()


# # pip install opencv-python
# # sudo apt install rename

# In[4]:


import shutil
import time
from PIL import Image
from tqdm import tqdm
from io import BytesIO
with capture.capture_output() as cap:
  get_ipython().run_line_magic('cd', '/home/super/Desktop/dreambooth/content')
  from smart_crop import *

#@markdown #Instance Images
#@markdown ----

#@markdown
#@markdown - Run the cell to upload the instance pictures.
#@markdown - You can add `external captions` in txt files by simply giving each txt file the same name as the instance image, for example dikgur (1).jpg and dikgur (1).txt, and upload them here, to use the external captions, check the box "external_captions" in the training cell. `All the images must have one same extension` jpg or png or....etc

Remove_existing_instance_images= True #@param{type: 'boolean'}
#@markdown - Uncheck the box to keep the existing instance images.

if Remove_existing_instance_images:
  if os.path.exists(str(INSTANCE_DIR)):
    get_ipython().system('rm -r "$INSTANCE_DIR"')
  if os.path.exists(str(CAPTIONS_DIR)):
    get_ipython().system('rm -r "$CAPTIONS_DIR"')

if not os.path.exists(str(INSTANCE_DIR)):
  get_ipython().run_line_magic('mkdir', '-p "$INSTANCE_DIR"')
if not os.path.exists(str(CAPTIONS_DIR)):
  get_ipython().run_line_magic('mkdir', '-p "$CAPTIONS_DIR"')

if os.path.exists(INSTANCE_DIR+"/.ipynb_checkpoints"):
  get_ipython().run_line_magic('rm', '-r $INSTANCE_DIR"/.ipynb_checkpoints"')


IMAGES_FOLDER_OPTIONAL="" #@param{type: 'string'}

#@markdown - If you prefer to specify directly the folder of the pictures instead of uploading, this will add the pictures to the existing (if any) instance images. Leave EMPTY to upload.

Smart_Crop_images= True #@param{type: 'boolean'}
Crop_size = 256 #@param ["512", "576", "640", "704", "768", "832", "896", "960", "1024"] {type:"raw"}

#@markdown - Smart crop the images without manual intervention.

while IMAGES_FOLDER_OPTIONAL !="" and not os.path.exists(str(IMAGES_FOLDER_OPTIONAL)):
  print('[1;31mThe image folder specified does not exist, use the colab file explorer to copy the path :')
  IMAGES_FOLDER_OPTIONAL=input('')

if IMAGES_FOLDER_OPTIONAL!="":
  if os.path.exists(IMAGES_FOLDER_OPTIONAL+"/.ipynb_checkpoints"):
    get_ipython().run_line_magic('rm', '-r "$IMAGES_FOLDER_OPTIONAL""/.ipynb_checkpoints"')

  with capture.capture_output() as cap:
    get_ipython().system('mv $IMAGES_FOLDER_OPTIONAL/*.txt $CAPTIONS_DIR')
  if Smart_Crop_images:
    for filename in tqdm(os.listdir(IMAGES_FOLDER_OPTIONAL), bar_format='  |{bar:15}| {n_fmt}/{total_fmt} Uploaded'):
      extension = filename.split(".")[-1]
      identifier=filename.split(".")[0]
      new_path_with_file = os.path.join(INSTANCE_DIR, filename)
      file = Image.open(IMAGES_FOLDER_OPTIONAL+"/"+filename)
      width, height = file.size
      if file.size !=(Crop_size, Crop_size):
        image=crop_image(file, Crop_size)
        if extension.upper()=="JPG" or extension.upper()=="jpg":
            image[0] = image[0].convert("RGB")
            image[0].save(new_path_with_file, format="JPEG", quality = 100)
        else:
            image[0].save(new_path_with_file, format=extension.upper())
      else:
        get_ipython().system('cp "$IMAGES_FOLDER_OPTIONAL/$filename" "$INSTANCE_DIR"')

  else:
    for filename in tqdm(os.listdir(IMAGES_FOLDER_OPTIONAL), bar_format='  |{bar:15}| {n_fmt}/{total_fmt} Uploaded'):
      get_ipython().run_line_magic('cp', '-r "$IMAGES_FOLDER_OPTIONAL/$filename" "$INSTANCE_DIR"')

  print('\n[1;32mDone, proceed to the next cell')


elif IMAGES_FOLDER_OPTIONAL =="":
  up=""
  uploaded = img_dict_list
  for filename in uploaded.keys():
    if filename.split(".")[-1]=="txt":
      shutil.copy(filename, CAPTIONS_DIR)
    up=[filename for filename in uploaded.keys() if filename.split(".")[-1]!="txt"]
  if Smart_Crop_images:
    for filename in tqdm(up, bar_format='  |{bar:15}| {n_fmt}/{total_fmt} Uploaded'):
      shutil.copy(filename, INSTANCE_DIR)
      extension = filename.split(".")[-1]
      identifier=filename.split(".")[0]
      new_path_with_file = os.path.join(INSTANCE_DIR, filename)
      file = Image.open(new_path_with_file)
      width, height = file.size
      if file.size !=(Crop_size, Crop_size):
        image=crop_image(file, Crop_size)
        if extension.upper()=="JPG" or extension.upper()=="jpg":
            image[0] = image[0].convert("RGB")
            image[0].save(new_path_with_file, format="JPEG", quality = 100)
        else:
            image[0].save(new_path_with_file, format=extension.upper())
      clear_output()
  else:
    for filename in tqdm(uploaded.keys(), bar_format='  |{bar:15}| {n_fmt}/{total_fmt} Uploaded'):
      shutil.copy(filename, INSTANCE_DIR)
      clear_output()
  print('\n[1;32mDone, proceed to the next cell')

with capture.capture_output() as cap:
  get_ipython().run_line_magic('cd', '"$INSTANCE_DIR"')
  get_ipython().system('find . -name "* *" -type f | rename \'s/ /-/g\'')
  get_ipython().run_line_magic('cd', '"$CAPTIONS_DIR"')
  get_ipython().system('find . -name "* *" -type f | rename \'s/ /-/g\'')
  
  get_ipython().run_line_magic('cd', '$SESSION_DIR')
  get_ipython().system('rm instance_images.zip captions.zip')
  get_ipython().system('zip -r instance_images instance_images')
  get_ipython().system('zip -r captions captions')
  get_ipython().run_line_magic('cd', '/home/super/Desktop/dreambooth/content')


# In[37]:


print(cap)


# !pip install diffusers
# !pip install git+https://github.com/huggingface/transformers
# !pip install torchvision
# !pip install bitsandbytes-cuda117

# In[8]:


#@markdown ---
#@markdown #Start DreamBooth
#@markdown ---
import os
from IPython.display import clear_output
from subprocess import getoutput
import time
import random

if os.path.exists(INSTANCE_DIR+"/.ipynb_checkpoints"):
  get_ipython().run_line_magic('rm', '-r $INSTANCE_DIR"/.ipynb_checkpoints"')

if os.path.exists(CONCEPT_DIR+"/.ipynb_checkpoints"):
  get_ipython().run_line_magic('rm', '-r $CONCEPT_DIR"/.ipynb_checkpoints"')

if os.path.exists(CAPTIONS_DIR+"/.ipynb_checkpoints"):
  get_ipython().run_line_magic('rm', '-r $CAPTIONS_DIR"/.ipynb_checkpoints"')

Resume_Training = False #@param {type:"boolean"}

if resume and not Resume_Training:
  print('[1;31mOverwrite your previously trained model ? answering "yes" will train a new model, answering "no" will resume the training of the previous model?  yes or no ?[0m')
  while True:
    ansres=input('')
    if ansres=='no':
      Resume_Training = True
      break
    elif ansres=='yes':
      Resume_Training = False
      resume= False
      break

while not Resume_Training and MODEL_NAME=="":
  print('[1;31mNo model found, use the "Model Download" cell to download a model.')
  time.sleep(5)

#@markdown  - If you're not satisfied with the result, check this box, run again the cell and it will continue training the current model.

MODELT_NAME=MODEL_NAME

UNet_Training_Steps=1500 #@param{type: 'number'}
UNet_Learning_Rate = 2e-6 #@param ["2e-5","1e-5","9e-6","8e-6","7e-6","6e-6","5e-6", "4e-6", "3e-6", "2e-6"] {type:"raw"}
untlr=UNet_Learning_Rate

#@markdown - These default settings are for a dataset of 10 pictures which is enough for training a face, start with 1500 or lower, test the model, if not enough, resume training for 200 steps, keep testing until you get the desired output, `set it to 0 to train only the text_encoder`.

Text_Encoder_Training_Steps=400 #@param{type: 'number'}

#@markdown - 200-450 steps is enough for a small dataset, keep this number small to avoid overfitting, set to 0 to disable, `set it to 0 before resuming training if it is already trained`.

Text_Encoder_Learning_Rate = 1e-6 #@param ["2e-6", "1e-6","8e-7","6e-7","5e-7","4e-7"] {type:"raw"}
txlr=Text_Encoder_Learning_Rate

#@markdown - Learning rate for both text_encoder and concept_text_encoder, keep it low to avoid overfitting (1e-6 is higher than 4e-7)

Text_Encoder_Concept_Training_Steps=0 #@param{type: 'number'}

#@markdown - Suitable for training a style/concept as it acts as heavy regularization, set it to 1500 steps for 200 concept images (you can go higher), set to 0 to disable, set both the settings above to 0 to fintune only the text_encoder on the concept, `set it to 0 before resuming training if it is already trained`.

trnonltxt=""
if UNet_Training_Steps==0:
   trnonltxt="--train_only_text_encoder"

Seed=''

ofstnse=""
Offset_Noise = False #@param {type:"boolean"}
#@markdown - Always use it for style training.

if Offset_Noise:
  ofstnse="--offset_noise"

External_Captions = False #@param {type:"boolean"}
#@markdown - Get the captions from a text file for each instance image.
extrnlcptn=""
if External_Captions:
  extrnlcptn="--external_captions"

Resolution = "256" #@param ["512", "576", "640", "704", "768", "832", "896", "960", "1024"]
Res=int(Resolution)

#@markdown - Higher resolution = Higher quality, make sure the instance images are cropped to this selected size (or larger).

fp16 = True

if Seed =='' or Seed=='0':
  Seed=random.randint(1, 999999)
else:
  Seed=int(Seed)

if fp16:
  prec="fp16"
else:
  prec="no"

precision=prec

resuming=""
if Resume_Training and os.path.exists(OUTPUT_DIR+'/unet/diffusion_pytorch_model.bin'):
  MODELT_NAME=OUTPUT_DIR
  print('[1;32mResuming Training...[0m')
  resuming="Yes"
elif Resume_Training and not os.path.exists(OUTPUT_DIR+'/unet/diffusion_pytorch_model.bin'):
  print('[1;31mPrevious model not found, training a new model...[0m')
  MODELT_NAME=MODEL_NAME
  while MODEL_NAME=="":
    print('[1;31mNo model found, use the "Model Download" cell to download a model.')
    time.sleep(5)

V2=False
if os.path.getsize(MODELT_NAME+"/text_encoder/pytorch_model.bin") > 670901463:
  V2=True

s = getoutput('nvidia-smi')
GCUNET="--gradient_checkpointing"
TexRes=Res
if Res<=768:
  GCUNET=""

if V2:  
  if Res>704:
    GCUNET="--gradient_checkpointing"
  if Res>576:
    TexRes=576

if 'A100' in s :
   GCUNET=""
   TexRes=Res


Enable_text_encoder_training= True
Enable_Text_Encoder_Concept_Training= True

if Text_Encoder_Training_Steps==0 :
   Enable_text_encoder_training= False
else:
  stptxt=Text_Encoder_Training_Steps

if Text_Encoder_Concept_Training_Steps==0:
   Enable_Text_Encoder_Concept_Training= False
else:
  stptxtc=Text_Encoder_Concept_Training_Steps

#@markdown ---------------------------
Save_Checkpoint_Every_n_Steps = False #@param {type:"boolean"}
Save_Checkpoint_Every=500 #@param{type: 'number'}
if Save_Checkpoint_Every==None:
  Save_Checkpoint_Every=1
#@markdown - Minimum 200 steps between each save.
stp=0
Start_saving_from_the_step=500 #@param{type: 'number'}
if Start_saving_from_the_step==None:
  Start_saving_from_the_step=0
if (Start_saving_from_the_step < 200):
  Start_saving_from_the_step=Save_Checkpoint_Every
stpsv=Start_saving_from_the_step
if Save_Checkpoint_Every_n_Steps:
  stp=Save_Checkpoint_Every
#@markdown - Start saving intermediary checkpoints from this step.

Disconnect_after_training=False #@param {type:"boolean"}

#@markdown - Auto-disconnect from google colab after the training to avoid wasting compute units.

def dump_only_textenc(trnonltxt, MODELT_NAME, INSTANCE_DIR, OUTPUT_DIR, PT, Seed, precision, Training_Steps):
    
    get_ipython().system('accelerate launch /home/super/Desktop/dreambooth/content/diffusers/examples/dreambooth/train_dreambooth.py     $trnonltxt     $extrnlcptn     $ofstnse     --image_captions_filename     --train_text_encoder     --dump_only_text_encoder     --pretrained_model_name_or_path="$MODELT_NAME"     --instance_data_dir="$INSTANCE_DIR"     --output_dir="$OUTPUT_DIR"     --captions_dir="$CAPTIONS_DIR"     --instance_prompt="$PT"     --seed=$Seed     --resolution=$TexRes     --mixed_precision=$precision     --train_batch_size=1     --gradient_accumulation_steps=1 --gradient_checkpointing     --use_8bit_adam     --learning_rate=$txlr     --lr_scheduler="linear"     --lr_warmup_steps=0     --max_train_steps=$Training_Steps')

def train_only_unet(stpsv, stp, SESSION_DIR, MODELT_NAME, INSTANCE_DIR, OUTPUT_DIR, PT, Seed, Res, precision, Training_Steps):
    clear_output()
    if resuming=="Yes":
      print('[1;32mResuming Training...[0m')
    print('[1;33mTraining the UNet...[0m')
    get_ipython().system('accelerate launch /home/super/Desktop/dreambooth/content/diffusers/examples/dreambooth/train_dreambooth.py     $extrnlcptn     $ofstnse     --image_captions_filename     --train_only_unet     --save_starting_step=$stpsv     --save_n_steps=$stp     --Session_dir=$SESSION_DIR     --pretrained_model_name_or_path="$MODELT_NAME"     --instance_data_dir="$INSTANCE_DIR"     --output_dir="$OUTPUT_DIR"     --captions_dir="$CAPTIONS_DIR"     --instance_prompt="$PT"     --seed=$Seed     --resolution=$Res     --mixed_precision=$precision     --train_batch_size=1     --gradient_accumulation_steps=1 $GCUNET     --use_8bit_adam     --learning_rate=$untlr     --lr_scheduler="linear"     --lr_warmup_steps=0     --max_train_steps=$Training_Steps')


if Enable_text_encoder_training :
  print('[1;33mTraining the text encoder...[0m')
  if os.path.exists(OUTPUT_DIR+'/'+'text_encoder_trained'):
    get_ipython().run_line_magic('rm', '-r $OUTPUT_DIR"/text_encoder_trained"')
  dump_only_textenc(trnonltxt, MODELT_NAME, INSTANCE_DIR, OUTPUT_DIR, PT, Seed, precision, Training_Steps=stptxt)

if Enable_Text_Encoder_Concept_Training:
  if os.path.exists(CONCEPT_DIR):
    if os.listdir(CONCEPT_DIR)!=[]:
      clear_output()
      if resuming=="Yes":
        print('[1;32mResuming Training...[0m')
      print('[1;33mTraining the text encoder on the concept...[0m')
      dump_only_textenc(trnonltxt, MODELT_NAME, CONCEPT_DIR, OUTPUT_DIR, PT, Seed, precision, Training_Steps=stptxtc)
    else:
      clear_output()
      if resuming=="Yes":
        print('[1;32mResuming Training...[0m')
      print('[1;31mNo concept images found, skipping concept training...')
      Text_Encoder_Concept_Training_Steps=0
      time.sleep(8)
  else:
      clear_output()
      if resuming=="Yes":
        print('[1;32mResuming Training...[0m')
      print('[1;31mNo concept images found, skipping concept training...')
      Text_Encoder_Concept_Training_Steps=0
      time.sleep(8)

if UNet_Training_Steps!=0:
  train_only_unet(stpsv, stp, SESSION_DIR, MODELT_NAME, INSTANCE_DIR, OUTPUT_DIR, PT, Seed, Res, precision, Training_Steps=UNet_Training_Steps)

if UNet_Training_Steps==0 and Text_Encoder_Concept_Training_Steps==0 and Text_Encoder_Training_Steps==0 :
  print('[1;32mNothing to do')
else:
  if os.path.exists('/home/super/Desktop/dreambooth/content/models/'+INSTANCE_NAME+'/unet/diffusion_pytorch_model.bin'):
    prc="--fp16" if precision=="fp16" else ""
    get_ipython().system('python3 /home/super/Desktop/dreambooth/content/diffusers/scripts/convertosdv2.py $prc $OUTPUT_DIR $SESSION_DIR/$Session_Name".ckpt"')
    clear_output()
    if os.path.exists(SESSION_DIR+"/"+INSTANCE_NAME+'.ckpt'):
      clear_output()
      print("[1;32mDONE, the CKPT model is in your Gdrive in the sessions folder")
      if Disconnect_after_training :
        time.sleep(20)
        runtime.unassign()
    else:
      print("[1;31mSomething went wrong")
  else:
    print("[1;31mSomething went wrong")


# 

#!/bin/bash

OC="/home/chinmay/Toolbox/oc"

POD=""  # Enter pod ID here
SOURCE_RESULTS_DIR="/workspace/Chinmay-Checkpoints-Ephemeral/HX4-PET-Translation"
TARGET_RESULTS_DIR="/home/chinmay/Desktop/Projects-Work/HX4-PET-Translation/Results/HX4-PET-Translation"
MODEL="hx4_pet_cyclegan_balanced"

mkdir $TARGET_RESULTS_DIR"/"$MODEL


# -------------------------
# Download final checkpoint

mkdir $TARGET_RESULTS_DIR"/"$MODEL"/checkpoints"

$OC cp $POD":"$SOURCE_RESULTS_DIR"/"$MODEL"/checkpoints/25000.pth" \
      $TARGET_RESULTS_DIR"/"$MODEL"/checkpoints/"


# ------------------
# Download train log

$OC cp $POD":"$SOURCE_RESULTS_DIR"/"$MODEL"/train_log.txt" \
      $TARGET_RESULTS_DIR"/"$MODEL"/"


# -----------------------------
# Download 'train' folder stuff

mkdir $TARGET_RESULTS_DIR"/"$MODEL"/train"

# Train config file
$OC cp $POD":"$SOURCE_RESULTS_DIR"/"$MODEL"/train/train_config.yaml" \
      $TARGET_RESULTS_DIR"/"$MODEL"/train/"

# PNG images
mkdir $TARGET_RESULTS_DIR"/"$MODEL"/train/images"

pix2pix_image_name_format="real_A1-real_A2-fake_B1-real_B1"
cyclegan_naive_image_name_format="real_A1-real_A2-fake_B1-rec_A1-rec_A2-real_B1-fake_A1-fake_A2-rec_B1"
cyclegan_balanced_image_name_format="real_A1-real_A2-fake_B1-fake_B2-rec_A1-rec_A2-real_B1-real_B2-fake_A1-fake_A2-rec_B1-rec_B2"

if [[ $MODEL == *"cyclegan_naive"* ]] 
    then
        image_name_format=$cyclegan_naive_image_name_format
elif [[ $MODEL == *"cyclegan_balanced"* ]] 
    then
        image_name_format=$cyclegan_balanced_image_name_format
else
    image_name_format=$pix2pix_image_name_format
fi

for train_step in {10000..60000..10000}
# for train_step in 10000 20000 25000
do 
    $OC cp $POD":"$SOURCE_RESULTS_DIR"/"$MODEL"/train/images/"$train_step"_"$image_name_format".png" \
          $TARGET_RESULTS_DIR"/"$MODEL"/train/images/"
done


# ---------------------------
# Download 'val' folder stuff

mkdir $TARGET_RESULTS_DIR"/"$MODEL"/val"

# Val config file
$OC cp $POD":"$SOURCE_RESULTS_DIR"/"$MODEL"/val/val_config.yaml" \
      $TARGET_RESULTS_DIR"/"$MODEL"/val/"

# PNG images
mkdir $TARGET_RESULTS_DIR"/"$MODEL"/val/images"
for train_step in {10000..60000..10000}
# for train_step in 10000 20000 25000
do 
    $OC cp $POD":"$SOURCE_RESULTS_DIR"/"$MODEL"/val/images/"$train_step"/" \
          $TARGET_RESULTS_DIR"/"$MODEL"/val/images/"$train_step"/"
done

# NRRD files
mkdir $TARGET_RESULTS_DIR"/"$MODEL"/val/saved"
for train_step in {10000..60000..10000}
# for train_step in 10000 20000 25000
do 
    $OC cp $POD":"$SOURCE_RESULTS_DIR"/"$MODEL"/val/saved/"$train_step"/" \
          $TARGET_RESULTS_DIR"/"$MODEL"/val/saved/"$train_step"/"
done


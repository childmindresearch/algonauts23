#!/bin/bash

# Download Algonauts23 challenge data
gdown 'https://drive.google.com/drive/folders/17RyBAnvDhrrt18Js2VZqSVi_nZ7bn3G3' --folder && \
    unzip 'algonauts_2023_challenge_data/subj*.zip' -d algonauts_2023_challenge_data && \
    rm algonauts_2023_challenge_data/subj*.zip

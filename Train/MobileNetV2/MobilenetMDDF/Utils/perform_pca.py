import joblib
import os
from tqdm import tqdm
import numpy as np

def perform_pca():

  feature_dir = 'Features/'
  ipca_path = feature_dir+'ipca/'

  # Load the PCA models
  ipca_conv = joblib.load(ipca_path+'ipca_conv.pkl')


  # dir of extracted features
  train_feature_path = 'Features/train/'

  p_train_feature_path = 'Features/p_train/'

  if not os.path.exists(p_train_feature_path):
    os.mkdir(p_train_feature_path)

  # iterate "train_feature_path" and perform pca on each batch of features and then saved the reducted feature in the ptrain dir
  for t in tqdm(os.listdir(train_feature_path)):

      train_features = np.load(train_feature_path+'/{}'.format(t))

      # select train features
      selected_conv10_features = train_features[:, :-1] #[:, :57600]
      train_targets = train_features[:, -1]

      # fit and transform train features
      reducted_conv10_features = ipca_conv.transform(selected_conv10_features)

      reducted_train_features = np.hstack((reducted_conv10_features, train_targets.reshape(-1, 1)))
      np.save(p_train_feature_path+t, reducted_train_features)




  # dir of extracted features
  val_feature_path = feature_dir+'val/'

  # create dir for saving reducted features
  p_val_feature_path = 'Features/p_val/'

  if not os.path.exists(p_val_feature_path):
    os.mkdir(p_val_feature_path)


  # iterate val feature path and perform pca on each batch of features and then saved the reducted feature in the p_val dir
  for t in tqdm(os.listdir(val_feature_path)):

      val_features = np.load(val_feature_path+'/{}'.format(t))

      # select val features
      selected_conv10_features = val_features[:, :-1]
      val_targets = val_features[:, -1]

      # fit and transform val features
      reducted_conv10_features = ipca_conv.transform(selected_conv10_features)

      reducted_val_features = np.hstack((reducted_conv10_features, val_targets.reshape(-1, 1)))
      np.save(p_val_feature_path+t, reducted_val_features)

  train_feature_set = os.listdir(p_train_feature_path)
  val_feature_set = os.listdir(p_val_feature_path)

  merged_train_features = []

  for idx, t in enumerate(train_feature_set):
      reducted_train_features = np.load(p_train_feature_path+'/{}'.format(t))
      merged_train_features.append(reducted_train_features)

  reducted_train_features = np.vstack(merged_train_features)

  merged_val_features = []
  for idx, t in enumerate(val_feature_set):
      reducted_val_features = np.load(p_val_feature_path+'/{}'.format(t))
      merged_val_features.append(reducted_val_features)

  reducted_val_features = np.vstack(merged_val_features)

  np.save('Features/mdff_mobilenetv3_train',reducted_train_features)
  np.save('Features/mdff_mobilenetv3_val',reducted_val_features)
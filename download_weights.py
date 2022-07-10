import gdown
import os
 
def download(google_drive_id, output):    
  if os.path.isfile(output):
    print('using cached', output)
  else:
    gdown.download(id=google_drive_id, output=output, quiet=False)

if __name__ == '__main__':
    os.makedirs('ckpt', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # DOWNLOAD MODEL WEIGHT
    download('1EWHWg9dUBFDd2OGvrxWOdTXU1Dhebb6z', 'ckpt/model.pt')

    # DOWNLOAD TEST FEATURE
    download('1NS9LPWz-UvIqHYUSrseac9tik_rkVHxz', 'data/test_features.pt')

    # DOWNLOAD TRAIN FEATURE
    download('1_igHhfI9kUTt2wklXmz3mqXAtrtzVvur', 'data/train_features.pt')
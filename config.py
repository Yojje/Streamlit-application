import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disable TF logging except errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
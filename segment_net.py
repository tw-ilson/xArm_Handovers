import segmentation_models_pytorch as smp

"""
This file will containt the image segmentation network used to pre-process the image obtained from wrist mounted camera. We will take a pre-trained network and tune it to produce a segmentation mask on the object if it is in frame.
"""

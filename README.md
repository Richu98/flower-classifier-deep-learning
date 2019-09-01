# Flower Classifier Deep Learning

PyTorch implementation of a Deep learning network to identify 102 different types of flowers (developed for the PyTorch Scholarship Challenge by UDACITY).

The used data set contains images of flowers from 102 different species divided in a training set and a validation set.The images can be downloaded here: https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip

**In addition, the repository contains a utility for testing the performance of a model** on the original flower dataset (http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz) or alternatively on a dataset obtained by downloading the first 10 resulting images from Google, querying it by the name of the flower categories. This allows to evaluate if the model is not affected by overfitting and if it is suitable for a real usecase.


- [TRAIN THE MODEL](#TRAIN-THE-MODEL)
- [TEST THE MODEL](#TEST-THE-MODEL)
- [PUBLISH THE MODEL RESULTS](#PUBLISH-THE-MODEL-RESULTS)
- [PUBLISH YOUR MODEL RESULTS](#PUBLISH-YOUR-MODEL-RESULTS)
- [THE TEST SET](#THE-TEST-SET)

# TRAIN THE MODEL

To train the model you can use different environments, including:

1. Google Colaboratory / Online notebook
2. Local machine

### Google Colaboratory / Online notebook

You can find here a basic flowers classifier implementation on a Google Colaboratory file: https://drive.google.com/open?id=11IXBEwk7Ms_bGWkCD0jxvyMGbY7u6_OX
This will allow you to train your model and save a checkpoint.

### Local machine


    $git clone https://github.com/Richu98/flower-classifier-deep-learning
    $cd flower-classifier-deep-learning



To train the model on the local computer, first install the dependencies:

    $pip install -r requirements.txt

after which you can simply run the training:

    python flower_classifier.py

The script will automatically download the necessary data sets.

# TEST THE MODEL

The colaboratory file https://drive.google.com/open?id=11IXBEwk7Ms_bGWkCD0jxvyMGbY7u6_OX already contains the code needed to test the model's performance.

To test a saved model, simply load it into memory and recall the function calc_accuracy:


    from test_model_pytorch_facebook_challenge import calc_accuracy

    model = load_model('classifier.pth')
    calc_accuracy(model, input_image_size=224)


The accuracy will be calculated on a dataset, downloaded on runtime, containing images classified by category, downloaded from Google search results.

You can also use the function to calculate the model's performance on a custom directory.

#### usage:
    calc_accuracy(model, input_image_size, use_google_testset=False, testset_path=None, batch_size=32,
                  norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225]))                        
    """
    Calculate the mean accuracy of the model on the test test
    :param model: the model
    :param use_google_testset: If true use the testset derived from google image
    :param testset_path: custom test set or if default google images dataset is downloaded
    :param batch_size:
    :param input_image_size:
    :param norm_mean: normalizazition mean for RGB channel
    :param norm_std: stardard deviation mean for RGB channel
    :return: the mean accuracy
    """


### From local machine


    $git clone https://github.com/Richu98/flower-classifier-deep-learning



# THE TEST SET

If you are just interested in the test set, you can download them here

Original test set: https://www.dropbox.com/s/da6ye9genbsdzbq/flower_data_original_test.zip?dl=1

Google test set: https://www.dropbox.com/s/3zmf1kq58o909rq/google_test_data.zip?dl=1

If you are using a notebook you can use it directly:



    !wget -O flower_data_orginal_test.zip "https://www.dropbox.com/s/da6ye9genbsdzbq/flower_data_original_test.zip?dl=1"
    !unzip flower_data_orginal_test.zip


    !wget -O google_test_data.zip "https://www.dropbox.com/s/3zmf1kq58o909rq/google_test_data.zip?dl=1"
    !unzip google_test_data.zip

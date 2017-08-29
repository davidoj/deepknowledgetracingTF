# Readme

This is a tensorflow implementation of an LSTM for predicting student test scores.

It is based on the "Deep Knowledge Tracing" work of [Piech (2015)](http://stanford.edu/%7Ecpiech/bio/papers/deepKnowledgeTracing.pdf) with elaborations by [Xiong et al (2016)](http://www.educationaldatamining.org/EDM2016/proceedings/paper_133.pdf) and [Khajah et al (2016)](https://www.cs.colorado.edu/%7Emozer/Research/Selected%20Publications/reprints/KhajahLindseyMozer2016.pdf)

## Usage

Requirements are python (I used 3.5, I imagine 2.7 should be fine but haven't tested), numpy and tensorflow.

You can run the model with the provided data (Assistments 2009-2010 data, with multiple entries and scaffolding questions removed) with

`python task.py --batch-size 16 --train-steps 5001`

Check the csv files for the data format if you would like to try your own data.

## Performance

With the provided test set, the net achieves an AUC of 0.78. This is signficantly worse than the result reported by Piech (0.86), but somewhat better than reported by Xiong (0.75). This version of the Assistments dataset may be somewhat more difficult than the original used by Piech.

One source of difference is that this model uses a trainable dense embedding layer to represent the vectors that encode skill ids and correct or incorrect responses. The network created by Xiong sends a raw one-hot encoding to the LSTM units, and also employs multiple LSTM layers. Piech uses a static embedding layer inspired by compressed sensing, and a single recurrent layer. The embedding layer was intended to help train the model using the 25 000 problem ids to encode the questions rather than the 200 odd skill ids to better distinguish between questions. I wasn't able to get particularly good performance by doing this, but this was trained on my laptop and it's possible that with a bit more power and a more reasonable training time, a network with good performance could be found.

The original dataset can be found [here](https://drive.google.com/file/d/0B3f_gAH-MpBmUmNJQ3RycGpJM0k/view?usp=sharing). Note that this contains some duplicated attempts tagged with different "skills".

It is likely that the fit could be improved with hyperparameter tuning, however it would be important to split out a validation set to tune the hyperparameters so that the test set fit is not exaggerated.

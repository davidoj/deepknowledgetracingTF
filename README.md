# Readme

This is a tensorflow implementation of an LSTM for predicting student test scores.

It is based on the "Deep Knowledge Tracing" work of (Piech (2015))[http://stanford.edu/%7Ecpiech/bio/papers/deepKnowledgeTracing.pdf] with elaborations by (Xiong et al (2016))[http://www.educationaldatamining.org/EDM2016/proceedings/paper_133.pdf] and (Khajah et al (2016))[https://www.cs.colorado.edu/%7Emozer/Research/Selected%20Publications/reprints/KhajahLindseyMozer2016.pdf]

## Usage

Requirements are numpy and tensorflow.

You can run the model with the provided data (Assistments 2009-2010 data, with multiple entries and scaffolding questions removed) with

`python task.py --batch-size 16 --train-steps 8001`

Check the csv files for the data format if you would like to try your own data.

## Performance

With the provided test set, the net achieves an AUC of 0.77-0.78. This is signficantly worse than the result reported by Piech (0.86), but somewhat better than reported by Xiong (0.75), and this version of the Assistments dataset may be somewhat more difficult than the original.

It is likely that the fit could be considerably improved with hyperparameter tuning, however it would be important to split out a validation set to tune the hyperparameters so that the test set fit is not over-exaggerated.
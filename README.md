# Gender-Recognition
Gender Recognition by Voice and Speech Analysis


This database was created to identify a voice as male or female, 
based upon acoustic properties of the voice and speech. 
The dataset consists of 3,168 recorded voice samples, collected from male and female speakers. 
The voice samples are pre-processed by acoustic analysis in R using the seewave and tuneR packages, 
with an analyzed frequency range of 0hz-280hz (human vocal range).


These files include a bunch of popular methods nowadays, such as Neural Network, SVM, 
and PCA (which is used to kill dimensions):

--start_up.R consists of 2 basic methods(baseline method and logistic regression)


The codes are all writen in Python(except start_up.R) and can be run directly without compile.


And I also uploaded some files which implemented certain existed famous packages like matplotlib, sklearn, etc.

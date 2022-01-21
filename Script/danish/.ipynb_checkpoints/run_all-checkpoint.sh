#!/bin/bash
# Run all the notebooks which we need to be timed on the AWS server for the computational runtimes table 
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute spliced_model_Exp.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute spliced_model_Gamma.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute spliced_model_Weibull.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute spliced_model_Lognormal.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute spliced_model_Inverse-Gaussian.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute spliced_model_Inverse-Weibull.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute spliced_model_Inverse-Gamma.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute spliced_model_Lomax.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute spliced_model_Log-Logistic.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute spliced_model_Burr.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute single_models.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute spliced_models_with_fixed_threshold.ipynb



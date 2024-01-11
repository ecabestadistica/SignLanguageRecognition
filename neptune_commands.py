

import neptune
import os
import neptune
import pandas as pd
import tensorflow as tf
import traceback

neptune_api_token = open("neptune_api_token.txt").read()
os.environ["NEPTUNE_API_TOKEN"] = neptune_api_token

min_run_id = 23 #included
top_run_id = 49 #included

for i in range(min_run_id, top_run_id+1):
    try:
        run = neptune.init_run(project="SignLanguageRecognition/AlphabetRecognition", with_id='AL-'+str(i))

        ## fetch model_params_count and train/epochs_executed
        # if any of both is missing, download and update them

        try:
            model_params_count = run["model_params_count"].fetch()
        except:
            model_params_count = -1

        try:
            epochs_executed = run["train/epochs_executed"].fetch()
        except:
            epochs_executed = -1

        if epochs_executed == -1:
            accuracy_dataframe = run["eval/accuracy"].fetch_values()
            # get num of epochs executed
            run["train/epochs_executed"] = len(accuracy_dataframe)

        if model_params_count == -1:

            model_name = run["parameters/base_model"].fetch()
            image_side = run["parameters/image_side"].fetch()
            accuracy = run["eval/accuracy"].fetch_last()

            ##round to 2 decimals
            accuracy = round(accuracy, 2)

            model_name = "TrainedModels/letters_new5_" + str(image_side) + "x" + str(image_side) + "_" + str(accuracy) + "pc_" + model_name + ".h5"

            # if file exists
            if not os.path.isfile(model_name):
                run["model_weights"].download(model_name)

            model = tf.keras.models.load_model(model_name)

            # get num of parameters
            if model_params_count == -1:
                run["model_params_count"] = model.count_params()

            del model

        run.stop()
    except Exception as e:
        print(traceback.format_exc())
        pass






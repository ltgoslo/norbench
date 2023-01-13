import argparse
import numpy as np
import pandas as pd
import data_preparation.data_preparation_sentiment as data_preparation_sentiment
import sentiment_finetuning
import pos_finetuning
import ner_finetuning
import fine_tuning
import utils.model_utils as model_utils
import utils.pos_utils as pos_utils


tasks = {
    'sentiment': {
        'train': sentiment_finetuning.train,
        'eval': sentiment_finetuning.test,
    },
    'pos': {
        'train': pos_finetuning.train,
        'eval': pos_finetuning.test,
    },
    'ner': {
        'train': ner_finetuning.train_use_eval,
        'test': ner_finetuning.test,
    }
}


def run_tasks(do_train, current_task, data_path, model_identifier, run_name, epochs, use_seqeval_evaluation_for_ner, use_class_weights_for_sent):
    
    if do_train == True:
        if current_task == 'ner':
            trainer = tasks[current_task]['train'](data_path, model_identifier, run_name, current_task, epochs, use_seqeval_evaluation_for_ner)
            test_results =  tasks[current_task]['test'](data_path, model_identifier, current_task, run_name, trainer)
            table = pd.DataFrame({
                            "Test F1": [test_results],
                            })

        else:
            if current_task == 'sentiment':
                training_object = tasks[current_task]['train'](data_path, short_model_name=model_identifier, epochs=epochs, use_class_weights=use_class_weights_for_sent, task=current_task)
            else:
                training_object = tasks[current_task]['train'](data_path, short_model_name=model_identifier, epochs=epochs, task=current_task)

            dev_score =  tasks[current_task]['eval'](data_path, "dev", short_model_name=model_identifier, task=current_task)

            test_score = tasks[current_task]['eval'](data_path, "test", short_model_name=model_identifier, task=current_task)

            table = pd.DataFrame({
                                "Dev F1": [dev_score],
                                "Test F1": [test_score]
                                })

        print(table)
        print(table.style.hide(axis='index').to_latex())
        table.to_csv(f"results/_{run_name}_{current_task}.tsv", sep="\t")
        print(f"Scores saved to results/_{run_name}_{current_task}.tsv")


def checking_data(path_to_data, task):
    return False if len(path_to_data) == 0 else True


def run_models_for_current_task(do_train, current_task, data_path, run_name, model_identifier, epochs, use_seqeval_evaluation_for_ner, use_class_weights_for_sent):

    if run_name == 'all':
            
            # if current model was not mentioned, then all of previously tested models have to be run
            
        run_all_models  = {model_path:model_ident for model_ident, model_path in model_utils.model_names.items()}
        for mod_path_bench, mod_ident_bench in run_all_models.items():
            run_tasks(do_train, current_task, data_path, mod_ident_bench, mod_path_bench, epochs, use_seqeval_evaluation_for_ner, use_class_weights_for_sent)
    else:
        run_tasks(do_train, current_task, data_path, model_identifier, run_name, epochs, use_seqeval_evaluation_for_ner, use_class_weights_for_sent)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="enter the name of the task: pos/ner/sentiment/all should be used", default='all')
    parser.add_argument("--path_to_dataset", help="path to the folder with data for current task", default="")
    parser.add_argument("--path_to_dataset_pos",  help="path to the folder with data for pos task if 'all' in task was used", default="")
    parser.add_argument("--path_to_dataset_ner", help="path to the folder with data for ner task if 'all' in task was used", default="")
    parser.add_argument("--path_to_dataset_sent", help="path to the folder with data for binary sentiment task if 'all' in task was used", default="")
    parser.add_argument("--model_name",  help="name of the model / 'all' to run all models that were tested for Norbench", default="")
    parser.add_argument("--path_to_model", help="path to model", default="ltgoslo/norbert")
    parser.add_argument("--download_cur_data", help="True if downloading of repositories with relevant data is needed",  type=bool, default=False)
    parser.add_argument("--do_train", help="True if model will be trained from scratch", type=bool, default=True)
    parser.add_argument("--batch_size", default=8)
    parser.add_argument("--learning_rate", default=2e-5)
    parser.add_argument("--use_class_weights_for_sent", action="store_true")
    parser.add_argument("--use_seqeval_evaluation_for_ner", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=10)

    args = parser.parse_args()

    if args.download_cur_data == True:
        data_path = args.download_cur_data
    else:
        data_path = args.path_to_dataset

    run_name = args.model_name
    model_identifier = args.path_to_model
    current_task = args.task
    epochs = args.epochs
    use_seqeval_evaluation_for_ner = args.use_seqeval_evaluation_for_ner
    use_class_weights_for_sent = args.use_class_weights_for_sent
    do_train = args.do_train
    

    if current_task == 'all':

        pathes_to_data = {'pos': args.path_to_dataset_pos,
                          'ner': args.path_to_dataset_ner,
                          'sentiment': args.path_to_dataset_sent}

        for tsk, path_tsk in pathes_to_data.items():
            if checking_data(path_tsk, tsk) == False:
                raise Exception(f'Check paths "{path_tsk}" to data for {tsk} task')
            else:
                run_models_for_current_task(do_train, tsk, path_tsk, run_name, model_identifier, epochs, use_seqeval_evaluation_for_ner, use_class_weights_for_sent)
    
    else:
        run_models_for_current_task(do_train, current_task, data_path, run_name, model_identifier, epochs, use_seqeval_evaluation_for_ner, use_class_weights_for_sent)
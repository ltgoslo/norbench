import argparse
import numpy as np
import pandas as pd
import os
import sentiment_finetuning
import ner_t5
import pos_finetuning
import ner_finetuning
import fine_tuning
from transformers import AutoModel
import utils.model_utils as model_utils
import utils.pos_utils as pos_utils
from distutils.util import strtobool

os.environ["WANDB_DISABLED"] = "true"

tasks = {
    'sentiment': {
        'train_evaluate': sentiment_finetuning.training_evaluating,
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

def_subtasks = {
    'sentiment': 'document_3',
    'pos': 'Bokmaal',
    'ner': 'Bokmaal'
}

def run_tasks(do_train, current_task, name_sub_info, data_path, model_identifier, run_name, epochs, use_seqeval_evaluation_for_ner, use_class_weights_for_sent):
    
    if name_sub_info == '':
        if data_path == True:
            name_sub_info = def_subtasks[current_task]
            print(f'...As subtask info was not mentioned, {name_sub_info} was chosen as default for task {current_task}...')
        else:
            name_sub_info = def_subtasks[current_task]
            print(f'...As subtask info was not mentioned and path_to_dataset was explicitly stated, {name_sub_info} was chosen as default for task {current_task}...')
            print(f'...If the current info for subtask is needed, it should be stated explicitly in argument --task_specific_info...')

    check_for_t5 = True if 't5' in AutoModel.from_pretrained(model_utils.get_full_model_names(model_identifier)).config.architectures[0].lower() else False

    if do_train == True:
        if current_task == 'ner':
            if check_for_t5 == False:
                trainer, tokenized_data = tasks[current_task]['train'](data_path, name_sub_info, model_identifier, run_name, current_task, epochs, use_seqeval_evaluation_for_ner)
                test_results =  tasks[current_task]['test'](data_path, name_sub_info, model_identifier, current_task, run_name, trainer=trainer, tokenized_data=tokenized_data)
            else:
                model, tokenizer = ner_t5.train_evaluation(data_path=data_path, sub_info=name_sub_info, model_name=model_identifier, run_name=run_name, task=current_task, epochs=epochs, use_seqeval_evaluation=use_seqeval_evaluation_for_ner)
                test_results = ner_t5.test(data_path=data_path, name_sub_info=name_sub_info, model_identifier=model_identifier, tokenizer=tokenizer, current_task=current_task, run_name=run_name)
            
            table = pd.DataFrame({
                    "Test F1": [test_results],
                                })

        else:
            # add iterations!!!!
            if current_task == 'sentiment':
                #dev_score, test_score = sentiment_finetuning.training_evaluating(task_specific_info, path_to_model, custom_wrapper, lr, max_length, batch_size, epochs)
                dev_score, test_score = tasks[current_task]['train_evaluate'](task_specific_info=name_sub_info, path_to_model_prev=model_identifier, custom_wrapper=False) 
            else:
                training_object = tasks[current_task]['train'](data_path, sub_task_info=name_sub_info, short_model_name=model_identifier, run_name=run_name, epochs=epochs, task=current_task)

                dev_score =  tasks[current_task]['eval'](data_path, "dev", sub_task_info=name_sub_info, short_model_name=model_identifier, run_name=run_name, task=current_task)

                test_score = tasks[current_task]['eval'](data_path, "test", sub_task_info=name_sub_info, short_model_name=model_identifier, run_name=run_name, task=current_task)

            table = pd.DataFrame({
                                "Dev F1": [dev_score],
                                "Test F1": [test_score]
                                })

        print(table)
        print(table.style.hide(axis='index').to_latex())

        if not os.path.exists("results"):
            os.makedirs("results")
            
        table.to_csv(f"results/{run_name}_{str(name_sub_info)}_{current_task}.tsv", sep="\t")
        print(f"Scores saved to results/{run_name}_{str(name_sub_info)}_{current_task}.tsv")


def checking_data(path_to_data, task):
    return False if len(path_to_data) == 0 else True


def run_models_for_current_task(do_train, current_task, name_sub_info, data_path, run_name, model_identifier, epochs, use_seqeval_evaluation_for_ner, use_class_weights_for_sent):

    if model_identifier == 'all':
        # if current model was not mentioned, then all of previously tested models have to be run
        run_all_models  = {model_path:model_ident for model_ident, model_path in model_utils.model_names.items()}
        for mod_path_bench, mod_ident_bench in run_all_models.items():
            run_tasks(do_train, current_task, name_sub_info, data_path, mod_ident_bench, mod_path_bench, epochs, use_seqeval_evaluation_for_ner, use_class_weights_for_sent)
    else:
        run_tasks(do_train, current_task, name_sub_info, data_path, model_identifier, run_name, epochs, use_seqeval_evaluation_for_ner, use_class_weights_for_sent)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="enter the name of the task: pos/ner/sentiment/all should be used", default='all')
    parser.add_argument("--task_specific_info", help="enter the name of the sub_task: name of language from original dataset / type of classification", default="")
    parser.add_argument("--path_to_dataset", help="If task is specified (pos/ner/sentiment): path to the folder with data for current task", default="")
    parser.add_argument("--path_to_dataset_pos",  help="If 'all' in task was used: path to the folder with data for pos task", default="")
    parser.add_argument("--path_to_dataset_ner", help="If 'all' in task was used: path to the folder with data for ner task", default="")
    parser.add_argument("--path_to_dataset_sent", help="If 'all' in task was used: path to the folder with data for binary sentiment task", default="")
    parser.add_argument("--model_name",  help="name of the model that will be used as an identifier for checkpoints", default="norbench_model")
    parser.add_argument("--path_to_model", help="path to model / 'all' to run all models that were tested for Norbench", default="ltgoslo/norbert")
    parser.add_argument("--download_cur_data", help="True if downloading of repositories with relevant data is needed",  type=bool, default=False)
    parser.add_argument("--do_train", help="True if model will be trained from scratch", type=bool, default=True)
    parser.add_argument("--batch_size", default=8)
    parser.add_argument("--learning_rate", default=2e-5)
    parser.add_argument("--class_balanced_for_sent", help="True if classes in sentiment analysis task should be balanced", choices=('True','False'), default='True')
    parser.add_argument("--use_seqeval_evaluation_for_ner", help="True if seqeval metrics should be used during evaluation for ner task", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=10)

    args = parser.parse_args()

    if args.download_cur_data == True:
        data_path = args.download_cur_data
    elif args.path_to_dataset == '' and args.download_cur_data == False:
        data_path = True
    else:
        data_path = args.path_to_dataset

    name_sub_info = args.task_specific_info

    run_name = args.model_name
    model_identifier = args.path_to_model
    current_task = args.task
    epochs = args.epochs
    use_seqeval_evaluation_for_ner = args.use_seqeval_evaluation_for_ner
    use_class_weights_for_sent = bool(strtobool(args.class_balanced_for_sent))
    do_train = args.do_train

    if current_task == 'all':

        pathes_to_data = {'sentiment': args.path_to_dataset_sent,
                         'ner': args.path_to_dataset_ner,
                         'pos': args.path_to_dataset_pos,
                         }

        for tsk, path_tsk in pathes_to_data.items():
            if checking_data(path_tsk, tsk) == False :
                print(f'...Path to data for {tsk} task was not mentioned. Path to relevant dataset with default subtask {def_subtasks[tsk]} was used. ...')
                path_tsk = True          
            run_models_for_current_task(do_train, tsk, name_sub_info, path_tsk, run_name, model_identifier, epochs, use_seqeval_evaluation_for_ner, use_class_weights_for_sent)
    
    else:
        run_models_for_current_task(do_train, current_task, name_sub_info, data_path, run_name, model_identifier, epochs, use_seqeval_evaluation_for_ner, use_class_weights_for_sent)